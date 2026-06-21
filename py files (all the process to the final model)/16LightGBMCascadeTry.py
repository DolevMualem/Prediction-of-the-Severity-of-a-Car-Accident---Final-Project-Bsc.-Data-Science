import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# ==========================================
# 1. טעינת והכנת הנתונים
# ==========================================
print("טוען נתונים למודל הדו-שלבי...")
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals.csv')

df = pd.read_csv(file_path)

target_col = 'collision_severity'

# מחיקת העמודות שהוסרו בגלל ערכי Null גבוהים (לפי העדכון החדש)
columns_to_drop = [target_col, 'lane_risk_group', 'age_of_vehicle']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X = df.drop(columns=columns_to_drop)
y = df[target_col] - 1  # 0: Fatal, 1: Serious, 2: Slight

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# פיצול נוסף של סט האימון עבור ה-Validation של Optuna
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# משימה 1: Slight (0) vs Major (1)
y_tr_stage1 = np.where(y_tr == 2, 0, 1)
y_val_stage1 = np.where(y_val == 2, 0, 1)

# משימה 2: Serious (0) vs Fatal (1)
major_indices_tr = (y_tr != 2)
X_tr_stage2 = X_tr[major_indices_tr]
y_tr_stage2 = np.where(y_tr[major_indices_tr] == 1, 0, 1)

major_indices_val = (y_val != 2)
X_val_stage2 = X_val[major_indices_val]
y_val_stage2 = np.where(y_val[major_indices_val] == 1, 0, 1)


# ==========================================
# 2. הגדרת פונקציות האופטימיזציה (Optuna)
# ==========================================

# פונקציית כוונון למודל שלב 1 (המסננת: האם זו תאונה קלה או חמורה?)
def objective_stage1(trial):
    params = {
        'is_unbalance': True,
        'n_estimators': trial.suggest_int('n_estimators', 150, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 70),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 300),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr_stage1)
    preds = model.predict(X_val)

    # מיקסום Macro Recall במקום F1 כדי לשים דגש על מניעת תאונות חמורות שמתפספסות
    return recall_score(y_val_stage1, preds, average='macro')


# פונקציית כוונון למודל שלב 2 (ההפרדה: קשה לעומת קטלני)
def objective_stage2(trial):
    params = {
        'is_unbalance': True,
        'n_estimators': trial.suggest_int('n_estimators', 150, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 150),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr_stage2, y_tr_stage2)
    preds = model.predict(X_val_stage2)

    return recall_score(y_val_stage2, preds, average='macro')


# ==========================================
# 3. הרצת האופטימיזציה
# ==========================================

print("\n>>> מתחיל כוונון אוטומטי למודל שלב 1 (Slight vs Major)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_stage1 = optuna.create_study(direction='maximize')
study_stage1.optimize(objective_stage1, n_trials=25)
print(f"הפרמטרים הטובים ביותר לשלב 1: {study_stage1.best_params}")

print("\n>>> מתחיל כוונון אוטומטי למודל שלב 2 (Serious vs Fatal)...")
study_stage2 = optuna.create_study(direction='maximize')
study_stage2.optimize(objective_stage2, n_trials=25)
print(f"הפרמטרים הטובים ביותר לשלב 2: {study_stage2.best_params}")

# ==========================================
# 4. אימון המודל הסופי עם הפרמטרים המנצחים
# ==========================================

print("\n>>> מאמן את המודלים הסופיים על כל דאטה האימון (Train)...")
y_train_stage1 = np.where(y_train == 2, 0, 1)
model_1_final = lgb.LGBMClassifier(is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
                                   **study_stage1.best_params)
model_1_final.fit(X_train, y_train_stage1)

major_indices_train = (y_train != 2)
y_train_stage2 = np.where(y_train[major_indices_train] == 1, 0, 1)
model_2_final = lgb.LGBMClassifier(is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
                                   **study_stage2.best_params)
model_2_final.fit(X_train[major_indices_train], y_train_stage2)

# ==========================================
# 5. בדיקה על סט המבחן הסופי (Test)
# ==========================================

print("\nמבצע תחזיות משולבות...")
preds_stage1 = model_1_final.predict(X_test)
final_predictions = np.full(y_test.shape, 2)  # ברירת מחדל: קל

# איפה ששלב 1 אמר שזו תאונה חמורה, מעבירים לשלב 2
major_test_indices = (preds_stage1 == 1)

if np.any(major_test_indices):
    preds_stage2 = model_2_final.predict(X_test[major_test_indices])
    # מיפוי חזרה: 0 בשלב 2 זה Serious (1), 1 בשלב 2 זה Fatal (0)
    preds_stage2_mapped = np.where(preds_stage2 == 0, 1, 0)
    final_predictions[major_test_indices] = preds_stage2_mapped

# חישוב ציון הריקול המיוחד
recalls = recall_score(y_test, final_predictions, average=None)
custom_recall_score = recalls.sum()

# ==========================================
# 6. תוצאות
# ==========================================

print("\n" + "=" * 50)
print("--- תוצאות מודל LightGBM דו-שלבי ---")
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