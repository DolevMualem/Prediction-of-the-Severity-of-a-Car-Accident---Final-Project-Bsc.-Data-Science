import pandas as pd
import numpy as np
import os
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests

# השתקת אזהרות מיותרות של SHAP
warnings.filterwarnings("ignore")

# ==========================================
# 1. טעינת הנתונים והכנתם
# ==========================================
print("Loading data...")
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals.csv')

df = pd.read_csv(file_path)

target_col = 'collision_severity'

# הסרת עמודות עם אחוז Null גבוה (בהתאם לעדכון האחרון שלך)
columns_to_drop = [target_col, 'lane_risk_group', 'age_of_vehicle']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X = df.drop(columns=columns_to_drop)

# === שינוי למודל בינארי ===
# 1 (Fatal) ו-2 (Serious) -> הופכים ל-1 (Major)
# 3 (Slight) -> הופך ל-0 (Slight)
y = np.where(df[target_col].isin([1, 2]), 1, 0)

# --- הנדסת מאפיינים חדשה לשיפור ה-Serious Recall ---
print("Creating advanced features...")

# 1. מדד אנרגיה בסיכון (מהירות מוכפלת בתנאי סביבה)
df['Risk_Energy_Index'] = df['speed_limit'] * (df['weather_conditions'] + df['road_surface_conditions'] + 1)

# 2. מדד צפיפות נפגעים (כמה אנשים נפגעו לכל רכב)
df['Casualty_Density'] = df['number_of_casualties'] / (df['number_of_vehicles'] + 0.1)

# 3. מדד פגיעות (סוג משתמש כפול מהירות)
df['Vulnerability_Speed_Index'] = df['casualty_class'] * df['speed_limit']

# פיצול לסט אימון ובדיקה (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# חישוב משקולות איזון דינמיות על בסיס סט האימון
print("Computing sample weights to handle class imbalance...")
sample_weights_train_full = compute_sample_weight(class_weight='balanced', y=y_train)

# ==========================================
# 2. אימון מודל ה-XGBoost (עם פרמטרי הזהב של Optuna + בינארי)
# ==========================================
print("\nTraining Final XGBoost Model with Optuna parameters (Binary)...")

xgb_params = {
    'objective': 'binary:logistic',  # שונה מ-multi:softprob לסיווג בינארי
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 161,
    'learning_rate': 0.06581489153812432,
    'max_depth': 9,
    'min_child_weight': 3,
    'max_delta_step': 7,
    'subsample': 0.6642445505775021,
    'colsample_bytree': 0.6412663096275449,
    'gamma': 2.94964507929579
}

final_xgb_model = xgb.XGBClassifier(**xgb_params)

# אימון המודל יחד עם המשקולות!
final_xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train_full)

# ==========================================
# 3. ביצוע תחזית (Inference) והערכת ביצועים
# ==========================================
print("\nPerforming Inference on Test Set...")
final_predictions = final_xgb_model.predict(X_test)

recalls = recall_score(y_test, final_predictions, average=None)
custom_recall_score = recalls.sum()

print("\n" + "=" * 50)
print("--- Final Binary XGBoost Results ---")
print("=" * 50)
print(f"Overall Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
print(f"⭐ Sum of Recalls: {custom_recall_score:.2f} ⭐\n")

print("Classification Report (0=Slight, 1=Major):")
print(classification_report(y_test, final_predictions))

print("\nFinal Confusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Slight (0)', 'Actual Major (1)'],
                     columns=['Pred Slight (0)', 'Pred Major (1)'])
print(cm_df)

# ==========================================
# 4. גרף חשיבות משתנים (Top 10 Feature Importance)
# ==========================================
print("\nGenerating Feature Importance Plot...")
importances = final_xgb_model.feature_importances_
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

plt.title('Top 10 Influential Features - Binary XGBoost\n(Normalized to 100%)', fontsize=16, fontweight='bold')
plt.xlabel('Contribution to Model Decision (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, fi_df['Importance_Pct'].max() + 5)
plt.tight_layout()
plt.show()

# ==========================================
# 5. מילון תרגום לממשק (UI Translation Dictionary)
# ==========================================
detailed_translation = {
    'weather_conditions': {'name': 'Weather', 'values': {0: 'Fine', 1: 'Rain/Snow', 2: 'High Winds', 3: 'Fog'}},
    'road_surface_conditions': {'name': 'Road Surface', 'values': {0: 'Dry', 1: 'Wet', 2: 'Snow/Ice'}},
    'speed_limit': {'name': 'Speed Limit', 'values': 'numeric'}
    # ניתן להוסיף את שאר המילון שלך כאן בדיוק כפי שהיה
}

# ==========================================
# 6. פונקציית פלט משולש למודל XGBoost הבינארי
# ==========================================
def predict_accident_binary_xgb(input_data_row):
    """
    מבצעת ניבוי על שורה בודדת מול מודל ה-XGBoost הבינארי.
    """
    probs = final_xgb_model.predict_proba(input_data_row)[0]
    pred_class = int(final_xgb_model.predict(input_data_row)[0])

    class_names = {0: "Slight", 1: "Major (Serious/Fatal)"}
    diagnosis = class_names[pred_class]
    confidence = probs[pred_class] * 100

    print(f"\nComputing SHAP explanation for class: {diagnosis}...")

    explainer = shap.TreeExplainer(final_xgb_model)
    shap_values = explainer.shap_values(input_data_row)

    # ב-XGBoost בינארי, SHAP מחזיר מערך דו ממדי יחיד של ההשפעות למחלקה החיובית
    current_shap_values = shap_values[0] if len(np.array(shap_values).shape) == 2 else shap_values

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
    explanation = f"The prediction was primarily influenced by the following factors: [{descriptions[0]}] and [{descriptions[1]}]."

    return diagnosis, strength, explanation


# ==========================================
# 7. חיבור API של מזג אוויר וסימולציה
# ==========================================
def get_live_weather_and_map_to_model(city_name, api_key):
    print(f"\nFetching live weather data for city: {city_name}...")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {'weather_conditions': 0, 'road_surface_conditions': 0}

        data = response.json()
        weather_main = data['weather'][0]['main'].lower()
        wind_speed = data['wind']['speed']

        weather_code = 2 if ('rain' in weather_main or 'snow' in weather_main) and wind_speed > 10 else 1 if ('rain' in weather_main or 'snow' in weather_main) else 3 if ('fog' in weather_main or 'mist' in weather_main) else 0
        road_code = 1 if 'rain' in weather_main else 2 if 'snow' in weather_main else 0

        print(f"API Weather: {weather_main.capitalize()} (Wind: {wind_speed}m/s) -> Mapped Weather Code: {weather_code}, Road Code: {road_code}")
        return {'weather_conditions': weather_code, 'road_surface_conditions': road_code}

    except Exception:
        return {'weather_conditions': 0, 'road_surface_conditions': 0}


print("\n" + "=" * 50)
print("--- END-TO-END SIMULATION: UI + API + MODEL ---")
print("=" * 50)

ui_input_df = X_test.iloc[[0]].copy()
user_city = "Tel Aviv,IL"
YOUR_REAL_API_KEY = "b245a156ed259736300c7d7e57ec0e4e"

live_weather_features = get_live_weather_and_map_to_model(user_city, YOUR_REAL_API_KEY)

for col, val in live_weather_features.items():
    if col in ui_input_df.columns:
        ui_input_df[col] = val

diag, strength, expl = predict_accident_binary_xgb(ui_input_df)

print("\n" + "X" * 70)
print("--- Final Output Sent Back to UI ---")
print(f"1. Diagnosis: {diag}")
print(f"2. Confidence: {strength}")
print(f"3. Explanation: {expl}")
print("X" * 70)