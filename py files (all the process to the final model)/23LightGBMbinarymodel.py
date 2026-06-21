import pandas as pd
import numpy as np
import os
import warnings
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

# === שלב הקידוד הבינארי ===
# 1 (Fatal) ו-2 (Serious) -> 1 (Major)
# 3 (Slight) -> 0 (Slight)
y = np.where(df[target_col].isin([1, 2]), 1, 0)

# פיצול לסט אימון ובדיקה (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# 2. אימון מודל LightGBM יחיד עם קנס מותאם
# ==========================================
print("\n--- Training Single LightGBM Model on FULL Dataset ---")

# הגדרת יחס הענישה: על כל טעות במחלקה 1 (Major), המודל ייקנס פי 1.5
# אתה יכול לשנות את זה ל-2.0 או 1.8 כדי להעלות עוד יותר את ה-Recall
CUSTOM_PENALTY = {0: 1.0, 1: 2}

model_binary = lgb.LGBMClassifier(
    class_weight=CUSTOM_PENALTY,  # הקנס שהצעת!
    n_estimators=350,  # יותר עצים כי יש לנו עכשיו הרבה יותר דאטה
    learning_rate=0.08,
    num_leaves=64,
    max_depth=7,
    min_child_samples=50,
    random_state=42,
    n_jobs=-1
)

model_binary.fit(X_train, y_train)
print("--- Training Complete ---")

# ==========================================
# 3. חיזוי (Inference) והערכת ביצועים
# ==========================================
print("\nPerforming Inference...")

# שימוש ישיר בפונקציית החיזוי (מכיוון שאיזנו באמצעות קנס, סף של 50% הוא מדויק)
final_predictions = model_binary.predict(X_test)

print("\n" + "=" * 50)
print(f"--- Final Binary Model Results (Penalty 1:{CUSTOM_PENALTY[1]}) ---")
print("=" * 50)
print(f"Overall Accuracy: {accuracy_score(y_test, final_predictions):.4f}\n")

print("Classification Report (0=Slight, 1=Major):")
print(classification_report(y_test, final_predictions))

print("\nFinal Confusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Slight (0)', 'Actual Major (1)'],
                     columns=['Pred Slight (0)', 'Pred Major (1)'])
print(cm_df)

# ==========================================
# 4. גרף חשיבות משתנים
# ==========================================
print("\nGenerating Feature Importance Chart...")
importances = model_binary.feature_importances_
importances_pct = (importances / importances.sum()) * 100

fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Pct': importances_pct
}).sort_values(by='Importance_Pct', ascending=False).head(10)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
ax = sns.barplot(x='Importance_Pct', y='Feature', data=fi_df, hue='Feature', palette="viridis", legend=False)

for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{width:.1f}%', (width, p.get_y() + p.get_height() / 2),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points',
                fontsize=11, fontweight='bold', color='black')

plt.title('Top 10 Influential Features (Single Binary Model)\n(Normalized to 100%)', fontsize=16, fontweight='bold')
plt.xlabel('Contribution to Model Decision (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, fi_df['Importance_Pct'].max() + 5)
plt.tight_layout()
plt.show()

# ==========================================
# 5. פונקציית הסבר מותאמת למודל יחיד (SHAP)
# ==========================================
detailed_translation = {
    'weather_conditions': {'name': 'Weather', 'values': {0: 'Fine', 1: 'Rain/Snow', 2: 'High Winds', 3: 'Fog'}},
    'road_surface_conditions': {'name': 'Road Surface', 'values': {0: 'Dry', 1: 'Wet', 2: 'Snow/Ice'}},
    'speed_limit': {'name': 'Speed Limit', 'values': 'numeric'}
}


def predict_accident_binary_single(input_data_row):
    """
    פונקציית הסבר למודל בינארי יחיד (נקייה ומהירה יותר)
    """
    probs = model_binary.predict_proba(input_data_row)[0]
    pred_class = 1 if probs[1] >= 0.5 else 0

    diagnosis = "Major (Serious/Fatal)" if pred_class == 1 else "Slight"
    confidence = probs[1] * 100 if pred_class == 1 else probs[0] * 100

    explainer = shap.TreeExplainer(model_binary)
    shap_values = explainer.shap_values(input_data_row)

    current_shap_values = shap_values[1][0] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values[
        0] if isinstance(shap_values, list) else shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

    feature_contributions = pd.DataFrame({
        'feature': X.columns,
        'impact_magnitude': np.abs(current_shap_values)
    }).sort_values(by='impact_magnitude', ascending=False)

    top_features = feature_contributions.head(2)['feature'].values

    descriptions = []
    for f_name in top_features:
        actual_value = input_data_row[f_name].values[0]
        if f_name in detailed_translation:
            field_info = detailed_translation[f_name]
            field_name_eng = field_info['name']
            if isinstance(field_info['values'], dict):
                val_eng = field_info['values'].get(actual_value, f"Code {actual_value}")
                descriptions.append(f"{field_name_eng}: {val_eng}")
            else:
                descriptions.append(f"{field_name_eng}: {actual_value}")
        else:
            descriptions.append(f"{f_name} ({actual_value})")

    strength = f"The accident was classified as '{diagnosis}' with {confidence:.1f}% confidence."
    explanation = f"Key influencing factors: [{descriptions[0]}] and [{descriptions[1]}]."

    return diagnosis, strength, explanation


# בדיקת סאניטי קטנה
sample_incident = X_test.iloc[[0]]
diag, strength, expl = predict_accident_binary_single(sample_incident)

print("\n" + "X" * 70)
print("--- End-to-End Prediction for Sample Case ---")
print(f"1. Diagnosis: {diag}")
print(f"2. Confidence: {strength}")
print(f"3. Explanation: {expl}")
print("X" * 70)