import pandas as pd
import numpy as np
import os
import warnings
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample  # משמש לביצוע ה-Downsampling
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests

warnings.filterwarnings("ignore",
                        message=".*LightGBM binary classifier with TreeExplainer shap values output has changed.*")

# ==========================================
# 1. טעינת הנתונים והכנתם
# ==========================================
print("Loading data...")
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals.csv')
df = pd.read_csv(file_path)

target_col = 'collision_severity'
X = df.drop(columns=[target_col])
y = df[target_col] - 1

# פיצול לסט אימון ובדיקה (70/30) - סט הבדיקה נשאר קדוש ולא נוגעים בו!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# 2. הכנת נתוני שלב 2 (Serious vs Fatal) - תמיד קבועים
# ==========================================
# שלב 2 עובד רק על המקרים החמורים (0 ו-1), אין צורך ב-Downsampling כי הם כבר מאוזנים יחסית
major_indices_train = (y_train != 2)
X_train_stage2 = X_train[major_indices_train]
y_train_stage2 = np.where(y_train[major_indices_train] == 1, 0, 1)

print("Training Stage 2 Model (Serious vs. Fatal)...")
model_stage2 = lgb.LGBMClassifier(
    is_unbalance=True, n_estimators=336, learning_rate=0.1333,
    num_leaves=40, max_depth=6, min_child_samples=98, random_state=42, n_jobs=-1
)
model_stage2.fit(X_train_stage2, y_train_stage2)

# ==========================================
# 3. מימוש ה-Downsampling המשולב ב-10 סיבובים (Ensemble) עבור שלב 1
# ==========================================
print("\n--- Starting Under-Bagging Ensemble (10 Iterations) for Stage 1 ---")

# נכין רשימה שתשמור את 10 המודלים שנאמן
ensemble_stage1_models = []
N_SPLITS = 10

# נפריד את נתוני האימון של שלב 1 לקבוצות לפי מחלקה כדי לבצע את החיתוך
y_train_stage1_actual = np.where(y_train == 2, 0, 1)  # 0 = Slight, 1 = Major (Serious/Fatal)

X_train_slight = X_train[y_train_stage1_actual == 0]
X_train_major = X_train[y_train_stage1_actual == 1]

# כמות הדגימות שנרצה לקחת מתוך מחלקת הרוב (Slight) בכל סיבוב, כדי להגיע לאיזון (למשל פי 1.5 או 1:1)
n_samples_slight = int(len(X_train_major) * 1.5)

for fold in range(N_SPLITS):
    print(f"Fold {fold + 1}/{N_SPLITS}: Performing Downsampling on 'Slight' class...")

    ### שורת ה-DOWNSAMPLING המפורשת! ###
    X_slight_downsampled = resample(
        X_train_slight,
        replace=False,
        n_samples=n_samples_slight,
        random_state=42 + fold  # שינוי ה-seed בכל סיבוב מבטיח דגימה של שורות קלות אחרות!
    )
    y_slight_downsampled = np.zeros(len(X_slight_downsampled))
    y_major_labels = np.ones(len(X_train_major))

    # חיבור הנתונים המאוזנים עבור הסיבוב הנוכחי
    X_fold_train = pd.concat([X_slight_downsampled, X_train_major])
    y_fold_train = np.concatenate([y_slight_downsampled, y_major_labels])

    # הגדרת מודל פולד ואימונו
    fold_model = lgb.LGBMClassifier(
        n_estimators=179, learning_rate=0.1264, num_leaves=64,
        max_depth=7, min_child_samples=85, random_state=42 + fold, n_jobs=-1
    )
    fold_model.fit(X_fold_train, y_fold_train)

    # שמירת המודל המאומן לאנסמבל
    ensemble_stage1_models.append(fold_model)

print("--- Ensemble Training Complete ---")

# ==========================================
# 4. ביצוע תחזית משולבת (Inference) עם ממוצע האנסמבל
# ==========================================
print("\nPerforming Inference using Ensemble Averaging...")

# א. נאסוף את תחזיות ההסתברות של שלב 1 מכל 10 המודלים
fold_probabilities = []
for model in ensemble_stage1_models:
    fold_probabilities.append(model.predict_proba(X_test)[:, 1])

# ב. נבצע ממוצע מתמטי על כל 10 הפולדים (זהו ה-Ensemble)
probs_stage1_avg = np.mean(fold_probabilities, axis=0)

# ג. החלת ספי ההחלטה (Thresholds)
THRESHOLD_STAGE1 = 0.45
THRESHOLD_STAGE2 = 0.35

preds_stage1 = np.where(probs_stage1_avg >= THRESHOLD_STAGE1, 1, 0)
final_predictions = np.full(y_test.shape, 2)
major_test_indices = (preds_stage1 == 1)

if np.any(major_test_indices):
    probs_stage2 = model_stage2.predict_proba(X_test[major_test_indices])[:, 1]
    preds_stage2 = np.where(probs_stage2 >= THRESHOLD_STAGE2, 1, 0)
    preds_stage2_mapped = np.where(preds_stage2 == 0, 1, 0)
    final_predictions[major_test_indices] = preds_stage2_mapped

# ==========================================
# 5. הצגת תוצאות והערכת ביצועים
# ==========================================
print("\n" + "=" * 50)
print("--- Final Cascade Ensemble LightGBM Results ---")
print("=" * 50)
print(f"Overall Accuracy: {accuracy_score(y_test, final_predictions):.4f}\n")
print("Classification Report (0=Fatal, 1=Serious, 2=Slight):")
print(classification_report(y_test, final_predictions))

print("\nFinal Confusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Fatal (0)', 'Actual Serious (1)', 'Actual Slight (2)'],
                     columns=['Pred Fatal (0)', 'Pred Serious (1)', 'Pred Slight (2)'])
print(cm_df)

# [שאר חלקי הקוד כמו SHAP וגרפים ו-API יעבדו כרגיל, רק ישתמשו ב-ensemble_stage1_models[0] או ממוצע לצורך התצוגה]