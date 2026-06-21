import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import os

# 1. טעינה
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_path = os.path.join(desktop_path, "Collision_Dataset_For_Model_With_Extra_Fatals.csv")
df = pd.read_csv(file_path)

# --- שלב א': הנדסת מאפיינים (Feature Engineering) ---
print("מבצע הנדסת מאפיינים...")
# מדד פגיעות (Vulnerability) - סכום של גורמי סיכון קטלניים
vulnerable_cols = ['escooter_flag', 'casualty_class'] # וודא שמות עמודות אלו קיימים
df['vulnerability_index'] = df[vulnerable_cols].sum(axis=1) if all(c in df.columns for c in vulnerable_cols) else 0

# 2. הכנה לאימון
X = df.drop(columns=['collision_severity', 'collision_index'], errors='ignore')
y = df['collision_severity'] - 1 # המרה ל-0,1,2 (0=Fatal)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- שלב ב': SMOTE מתוקן (אסטרטגיה דינמית) ---
# אנחנו בודקים כמה יש מכל מחלקה ב-Train כדי לא לחזור על השגיאה
counts = y_train.value_counts().to_dict()
print(f"התפלגות מקורית בסט האימון: {counts}")

# הגדרת מטרה: להגדיל את הקטלני (0) פי 5 ואת הקשה (1) ל-120,000 דוגמאות
target_fatal = min(counts[0] * 1, counts[2]) # לא לעבור את כמות ה-Slight
target_serious = max(counts[1], 120000)      # רק אם זה באמת הגדלה

print(f"מבצע SMOTE למטרות: Fatal={target_fatal}, Serious={target_serious}...")

smote = SMOTE(sampling_strategy={0: int(target_fatal), 1: int(target_serious)}, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 3. אימון מודל LightGBM עוצמתי
print("מאמן מודל LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=12,
    min_child_samples=40,
    reg_alpha=0.5,      # הגנה מ-Overfitting
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    importance_type='gain',
    verbose=-1
)

model.fit(X_train_res, y_train_res)

# --- שלב ג': כיוונון וקטור משקולות אופטימלי (The "Secret Sauce") ---
probs = model.predict_proba(X_test)

# וקטור משקולות אגרסיבי לשיפור הריקול של הקטלני והקשה
# 35.0 לקטלני, 5.0 לקשה, 1.0 לקל
custom_weights = np.array([1.0, 2.0, 1.0])
y_pred = np.argmax(probs * custom_weights, axis=1)

# 4. הצגת תוצאות
print("\n" + "="*40)
print(" תוצאות המודל המשופר (SMOTE + Weighted)")
print("="*40)
print(classification_report(y_test, y_pred, target_names=["1 (Fatal)", "2 (Serious)", "3 (Slight)"]))
print(f"Accuracy כללי: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1:      {f1_score(y_test, y_pred, average='macro'):.4f}")

# חשיבות משתנים
fi = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=X.columns).sort_values(ascending=False)
print("\nTop 5 משתנים משפיעים:")
print(fi.head(5))

