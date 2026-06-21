import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

# ==========================================
# 1. טעינת הנתונים
# ==========================================
print("טוען נתונים לאנסמבל...")
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals.csv')

df = pd.read_csv(file_path)

target_col = 'collision_severity'

columns_to_drop = [target_col, 'lane_risk_group', 'age_of_vehicle']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X = df.drop(columns=columns_to_drop)
y = df[target_col] - 1  # 0: Fatal, 1: Serious, 2: Slight

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

sample_weights_train_full = compute_sample_weight(class_weight='balanced', y=y_train)

# ==========================================
# 2. אימון המודלים (עם הפרמטרים המנצחים שלך!)
# ==========================================
print("\n>>> מאמן את XGBoost (חד שלבי)...")
xgb_params = {
    'objective': 'multi:softprob',  # חשוב! השתנה מ-softmax כדי לקבל אחוזים
    'num_class': 3,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 161,
    'learning_rate': 0.0658,
    'max_depth': 9,
    'min_child_weight': 3,
    'max_delta_step': 7,
    'subsample': 0.664,
    'colsample_bytree': 0.641,
    'gamma': 2.95
}
model_xgb = xgb.XGBClassifier(**xgb_params)
model_xgb.fit(X_train, y_train, sample_weight=sample_weights_train_full)

print(">>> מאמן את LightGBM (דו-שלבי)...")
# שלב 1 של LGBM (קל מול קשה/קטלני)
y_train_stage1 = np.where(y_train == 2, 0, 1)  # 0=Slight, 1=Major
lgb_params_1 = {
    'is_unbalance': True,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'n_estimators': 179,
    'learning_rate': 0.126,
    'num_leaves': 64,
    'max_depth': 7,
    'min_child_samples': 85
}
model_lgb_1 = lgb.LGBMClassifier(**lgb_params_1)
model_lgb_1.fit(X_train, y_train_stage1)

# שלב 2 של LGBM (קשה מול קטלני)
major_indices_train = (y_train != 2)
y_train_stage2 = np.where(y_train[major_indices_train] == 1, 0, 1)  # 0=Serious, 1=Fatal
lgb_params_2 = {
    'is_unbalance': True,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'n_estimators': 336,
    'learning_rate': 0.133,
    'num_leaves': 40,
    'max_depth': 6,
    'min_child_samples': 98
}
model_lgb_2 = lgb.LGBMClassifier(**lgb_params_2)
model_lgb_2.fit(X_train[major_indices_train], y_train_stage2)

# ==========================================
# 3. בניית ה- Soft Voting Ensemble
# ==========================================
print("\n>>> יוצר תחזיות אנסמבל על קבוצת הבדיקה (Test)...")

# --- תחזיות הסתברות של XGBoost ---
# מקבלים מטריצה של הסתברויות עבור כל מחלקה [Prob_Fatal, Prob_Serious, Prob_Slight]
xgb_probs = model_xgb.predict_proba(X_test)

# --- תחזיות הסתברות של LightGBM הדו-שלבי ---
lgb_probs = np.zeros((X_test.shape[0], 3))

# שלב 1: הסתברות לקלה (0) מול חמורה (1)
lgb_stage1_probs = model_lgb_1.predict_proba(X_test)

for i in range(len(X_test)):
    prob_slight = lgb_stage1_probs[i][0]
    prob_major = lgb_stage1_probs[i][1]

    # שלב 2: אם המודל חושב שזה Major, נבדוק כמה הוא בטוח אם זה Fatal או Serious
    stage2_prob = model_lgb_2.predict_proba(X_test.iloc[[i]])[0]
    prob_serious_given_major = stage2_prob[0]
    prob_fatal_given_major = stage2_prob[1]

    # חוק בייס (Bayes): הסתברות לקטלני = ההסתברות לחמור כפול ההסתברות לקטלני מתוך חמור
    lgb_probs[i][0] = prob_major * prob_fatal_given_major  # Fatal
    lgb_probs[i][1] = prob_major * prob_serious_given_major  # Serious
    lgb_probs[i][2] = prob_slight  # Slight

# --- שילוב (Voting) ---
# אנחנו נותנים משקל שווה לכל מודל (50/50)
ensemble_probs = (xgb_probs * 0.5) + (lgb_probs * 0.5)

# ההחלטה הסופית: המחלקה שקיבלה את ההסתברות המשוקללת הכי גבוהה
final_predictions = np.argmax(ensemble_probs, axis=1)

# חישוב סכום הריקולים
recalls = recall_score(y_test, final_predictions, average=None)
custom_recall_score = recalls.sum()

# ==========================================
# 4. תוצאות האנסמבל
# ==========================================
print("\n" + "=" * 50)
print("🏆 --- תוצאות מודל האנסמבל (XGBoost + LightGBM) --- 🏆")
print("=" * 50)
print(f"Accuracy הכללי: {accuracy_score(y_test, final_predictions):.4f}")
print(f"⭐ הריקול המשוקלל (סכום שלושת הריקולים): {custom_recall_score:.2f} ⭐\n")

print("Classification Report (0=Fatal, 1=Serious, 2=Slight):")
print(classification_report(y_test, final_predictions))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Fatal (0)', 'Actual Serious (1)', 'Actual Slight (2)'],
                     columns=['Pred Fatal (0)', 'Pred Serious (1)', 'Pred Slight (2)'])
print(cm_df)