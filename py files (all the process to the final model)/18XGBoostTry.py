import pandas as pd
import numpy as np
import os
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

# ==========================================
# 1. טעינת הנתונים והכנתם
# ==========================================
print("טוען נתונים...")
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals.csv')

df = pd.read_csv(file_path)

# עמודת המטרה: 0=Fatal, 1=Serious, 2=Slight
target_col = 'collision_severity'
X = df.drop(columns=[target_col])
y = df[target_col] - 1

# פיצול לסט אימון ובדיקה (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# פיצול נוסף של סט האימון עבור ה-Validation של Optuna
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# ==========================================
# חישוב משקולות פעם אחת בלבד
# ==========================================
print("מחשב משקולות איזון למחלקות...")
sample_weights_tr = compute_sample_weight(class_weight='balanced', y=y_tr)
sample_weights_train_full = compute_sample_weight(class_weight='balanced', y=y_train)

# ==========================================
# 2. פונקציית האופטימיזציה של Optuna עבור XGBoost
# ==========================================
def objective_xgb(trial):
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 150, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
    preds = model.predict(X_val)

    # ממקסמים את ה-Macro Recall
    return recall_score(y_val, preds, average='macro')


# ==========================================
# 3. הרצת האופטימיזציה
# ==========================================
print(">>> מתחיל כוונון אוטומטי למודל XGBoost (ממוקד בשיפור ה-Recall המשותף)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=25)

print("\n" + "=" * 50)
print("🏆 הפרמטרים המנצחים ש-Optuna בחר עבור XGBoost 🏆")
for key, value in study_xgb.best_params.items():
    print(f"  {key}: {value}")
print("=" * 50)

# ==========================================
# 4. אימון המודל הסופי ובדיקה
# ==========================================
print("\nמאמן את מודל ה-XGBoost הסופי על כל סט האימון עם הפרמטרים המנצחים...")

final_params = study_xgb.best_params
final_params['objective'] = 'multi:softmax'
final_params['num_class'] = 3
final_params['tree_method'] = 'hist'
final_params['random_state'] = 42
final_params['n_jobs'] = -1

final_xgb_model = xgb.XGBClassifier(**final_params)
final_xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train_full)

print("\nמבצע תחזיות על סט הבדיקה (Test)...")
final_predictions = final_xgb_model.predict(X_test)

recalls = recall_score(y_test, final_predictions, average=None)
custom_recall_score = recalls.sum()

# ==========================================
# 5. תוצאות
# ==========================================
print("\n--- תוצאות מודל XGBoost לאחר Optuna ---")
print(f"Accuracy הכללי: {accuracy_score(y_test, final_predictions):.4f}")
print(f"⭐ הריקול המשוקלל (סכום): {custom_recall_score:.2f} ⭐\n")

print("Classification Report (0=Fatal, 1=Serious, 2=Slight):")
print(classification_report(y_test, final_predictions))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Fatal (0)', 'Actual Serious (1)', 'Actual Slight (2)'],
                     columns=['Pred Fatal (0)', 'Pred Serious (1)', 'Pred Slight (2)'])
print(cm_df)


# ==========================================
# 6. חשיבות מאפיינים (Top 10 Feature Importance)
# ==========================================
print("\n" + "=" * 50)
print("🌟 10 המאפיינים המשפיעים ביותר על המודל 🌟")
print("=" * 50)

# חילוץ המשקלים מתוך המודל המאומן
importances = final_xgb_model.feature_importances_
feature_names = X.columns

# יצירת טבלה ומיון מהגבוה לנמוך
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)

# נרמול לאחוזים להצגה קריאה יותר
fi_df['Importance (%)'] = fi_df['Importance'] * 100

# הדפסה טקסטואלית יפה
for index, row in fi_df.iterrows():
    print(f"{row['Feature']:<30} | {row['Importance (%)']:>5.2f}%")

# ציור הגרף
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance (%)', y='Feature', data=fi_df, palette='viridis')
plt.title('Top 10 Feature Importances - XGBoost', fontsize=16)
plt.xlabel('Importance (%)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()