import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. טעינת הנתונים והכנתם
# ==========================================
print("טוען נתונים...")
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model.csv')

df = pd.read_csv(file_path)

# עמודת המטרה: 0=Fatal, 1=Serious, 2=Slight
target_col = 'collision_severity'
X = df.drop(columns=[target_col])
y = df[target_col] - 1

# פיצול לסט אימון ובדיקה (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# 2. הכנת הנתונים לגישת ה-Cascade (מודל דו-שלבי)
# ==========================================

# --- משימה 1: קל (Slight - 0) לעומת חמור (Major - 1) ---
y_train_stage1 = np.where(y_train == 2, 0, 1)

# --- משימה 2: קשה (Serious - 0) לעומת קטלני (Fatal - 1) ---
major_indices_train = (y_train != 2)
X_train_stage2 = X_train[major_indices_train]
y_train_stage2 = np.where(y_train[major_indices_train] == 1, 0, 1)

# ==========================================
# 3. אימון המודלים (עם פרמטרי הזהב של Optuna)
# ==========================================

print("\nמאמן את מודל שלב 1 (Slight vs. Major)...")
model_stage1 = lgb.LGBMClassifier(
    is_unbalance=True,
    n_estimators=213,
    learning_rate=0.0105973995766038,
    num_leaves=22,
    max_depth=6,
    min_child_samples=289,
    random_state=42,
    n_jobs=-1
)
model_stage1.fit(X_train, y_train_stage1)

print("מאמן את מודל שלב 2 (Serious vs. Fatal)...")
model_stage2 = lgb.LGBMClassifier(
    is_unbalance=True,
    n_estimators=177,
    learning_rate=0.010054714420590874,
    num_leaves=23,
    max_depth=4,
    min_child_samples=112,
    random_state=42,
    n_jobs=-1
)
model_stage2.fit(X_train_stage2, y_train_stage2)

# ==========================================
# 4. ביצוע תחזית משולבת (Inference) על סט הבדיקה
# ==========================================
print("\nמבצע תחזיות...")

# א. תחזית שלב 1 (סינון ראשוני)
preds_stage1 = model_stage1.predict(X_test)

# ב. אתחול כלל התחזיות כ'קלות' (2)
final_predictions = np.full(y_test.shape, 2)

# ג. מציאת האירועים שהמודל הראשון סיווג כחמורים
major_test_indices = (preds_stage1 == 1)

# ד. העברת האירועים החמורים להכרעה במודל השני
if np.any(major_test_indices):
    preds_stage2 = model_stage2.predict(X_test[major_test_indices])

    # מיפוי התוצאות חזרה לקידוד המקורי: 0 של מודל 2 הופך ל-1 (Serious), ו-1 הופך ל-0 (Fatal)
    preds_stage2_mapped = np.where(preds_stage2 == 0, 1, 0)

    # הכנסת התוצאות לתוך המערך הסופי
    final_predictions[major_test_indices] = preds_stage2_mapped

# ==========================================
# 5. הצגת תוצאות והערכת ביצועים
# ==========================================
print("\n" + "=" * 50)
print("--- תוצאות המודל הסופי (Cascade LightGBM) ---")
print("=" * 50)
print(f"Accuracy הכללי: {accuracy_score(y_test, final_predictions):.4f}\n")

print("Classification Report (0=Fatal, 1=Serious, 2=Slight):")
print(classification_report(y_test, final_predictions))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Fatal (0)', 'Actual Serious (1)', 'Actual Slight (2)'],
                     columns=['Pred Fatal (0)', 'Pred Serious (1)', 'Pred Slight (2)'])
print(cm_df)