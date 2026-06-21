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

# השתקת אזהרת ה-SHAP הספציפית עבור LightGBM כדי לשמור על טרמינל נקי
warnings.filterwarnings("ignore",
                        message=".*LightGBM binary classifier with TreeExplainer shap values output has changed.*")

# ==========================================
# 1. טעינת הנתונים והכנתם
# ==========================================
print("Loading data...")
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals.csv')

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
# 3. אימון המודלים (עם פרמטרי הזהב המעודכנים של Optuna)
# ==========================================

print("\nTraining Stage 1 Model (Slight vs. Major) with UPDATED Optuna parameters...")
model_stage1 = lgb.LGBMClassifier(
    is_unbalance=True,
    n_estimators=179,
    learning_rate=0.12644586230081986,
    num_leaves=64,
    max_depth=7,
    min_child_samples=85,
    random_state=42,
    n_jobs=-1
)
model_stage1.fit(X_train, y_train_stage1)

print("Training Stage 2 Model (Serious vs. Fatal) with UPDATED Optuna parameters...")
model_stage2 = lgb.LGBMClassifier(
    is_unbalance=True,
    n_estimators=336,
    learning_rate=0.13334762065891254,
    num_leaves=40,
    max_depth=6,
    min_child_samples=98,
    random_state=42,
    n_jobs=-1
)
model_stage2.fit(X_train_stage2, y_train_stage2)

# ==========================================
# 4. ביצוע תחזית משולבת (Inference) על סט הבדיקה
# ==========================================
print("\nPerforming Cascade Inference on Test Set...")

# א. תחזית שלב 1 (סינון ראשוני)
preds_stage1 = model_stage1.predict(X_test)

# ב. אתחול כלל התחזיות כ'קלות' (2)
final_predictions = np.full(y_test.shape, 2)

# ג. מציאת האירועים שהמודל הראשון סיווג כחמורים
major_test_indices = (preds_stage1 == 1)

# ד. העברת האירועים החמורים להכרעה במודל השני
if np.any(major_test_indices):
    preds_stage2 = model_stage2.predict(X_test[major_test_indices])

    # מיפוי התוצאות חזרה לקידוד המקורי:
    # מודל 2 חזה 0 -> נהפוך ל-1 (Serious)
    # מודל 2 חזה 1 -> נהפוך ל-0 (Fatal)
    preds_stage2_mapped = np.where(preds_stage2 == 0, 1, 0)

    # הכנסת התוצאות לתוך המערך הסופי במקומות הנכונים
    final_predictions[major_test_indices] = preds_stage2_mapped

# ==========================================
# 5. הצגת תוצאות והערכת ביצועים
# ==========================================
print("\n" + "=" * 50)
print("--- Final Cascade LightGBM Results ---")
print("=" * 50)
print(f"Overall Accuracy: {accuracy_score(y_test, final_predictions):.4f}\n")

print("Classification Report (0=Fatal, 1=Serious, 2=Slight):")
print(classification_report(y_test, final_predictions))

print("\nFinal Confusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Fatal (0)', 'Actual Serious (1)', 'Actual Slight (2)'],
                     columns=['Pred Fatal (0)', 'Pred Serious (1)', 'Pred Slight (2)'])
print(cm_df)

# ==========================================
# 6. גרפי חשיבות משתנים (עבור שני המודלים)
# ==========================================

# --- גרף מודל שלב 1 (כחול) ---
print("\nTop 10 Most Influential Features (Stage 1 Model: Slight vs Major):")
importances_stage1 = model_stage1.feature_importances_
importances_pct_1 = (importances_stage1 / importances_stage1.sum()) * 100

fi_df_1 = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Pct': importances_pct_1
}).sort_values(by='Importance_Pct', ascending=False).head(10)

print(fi_df_1)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
ax1 = sns.barplot(x='Importance_Pct', y='Feature', data=fi_df_1, hue='Feature', palette="Blues_r", legend=False)

for p in ax1.patches:
    width = p.get_width()
    ax1.annotate(f'{width:.1f}%', (width, p.get_y() + p.get_height() / 2),
                 ha='left', va='center', xytext=(5, 0), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='black')

plt.title('Top 10 Influential Features (Stage 1: Slight vs. Major)\n(Normalized to 100%)', fontsize=16,
          fontweight='bold')
plt.xlabel('Contribution to Model Decision (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, fi_df_1['Importance_Pct'].max() + 5)
plt.tight_layout()
plt.show()

# --- גרף מודל שלב 2 (אדום) ---
print("\nTop 10 Most Influential Features (Stage 2 Model: Serious vs Fatal):")
importances_stage2 = model_stage2.feature_importances_
importances_pct_2 = (importances_stage2 / importances_stage2.sum()) * 100

fi_df_2 = pd.DataFrame({
    'Feature': X_train_stage2.columns,
    'Importance_Pct': importances_pct_2
}).sort_values(by='Importance_Pct', ascending=False).head(10)

print(fi_df_2)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
# שימוש בצבעים אדומים כדי להבדיל שזהו מודל התאונות החמורות
ax2 = sns.barplot(x='Importance_Pct', y='Feature', data=fi_df_2, hue='Feature', palette="Reds_r", legend=False)

for p in ax2.patches:
    width = p.get_width()
    ax2.annotate(f'{width:.1f}%', (width, p.get_y() + p.get_height() / 2),
                 ha='left', va='center', xytext=(5, 0), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='black')

plt.title('Top 10 Influential Features (Stage 2: Serious vs. Fatal)\n(Normalized to 100%)', fontsize=16,
          fontweight='bold')
plt.xlabel('Contribution to Model Decision (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, fi_df_2['Importance_Pct'].max() + 5)
plt.tight_layout()
plt.show()

# ==========================================
# 7. יישום פלט משולש (Triple Output) למודל Cascade + מילון
# ==========================================

detailed_translation = {
    'number_of_vehicles': {'name': 'Number of vehicles involved', 'values': 'numeric'},
    'number_of_casualties': {'name': 'Number of casualties', 'values': 'numeric'},
    'road_type': {'name': 'Road type',
                  'values': {0: 'Roundabout', 1: 'One way street', 2: 'Slip Road', 3: 'Dual carriageway',
                             4: 'Single carriageway'}},
    'speed_limit': {'name': 'Speed limit', 'values': 'numeric'},
    'junction_detail': {'name': 'Junction detail',
                        'values': {0: 'Not at junction or within 20 metres', 1: 'Using private drive or entrance',
                                   2: 'Other Junction', 3: 'T or staggered junction',
                                   4: 'Crossroads / Junction with >4 arms'}},
    'pedestrian_crossing': {'name': 'Pedestrian crossing',
                            'values': {0: 'No physical crossing facility within 50m', 1: 'Central refuge',
                                       2: 'Zebra crossing', 3: 'Pedestrian light crossing / signal',
                                       4: 'Human crossing control / Footbridge or subway'}},
    'light_conditions': {'name': 'Light conditions', 'values': {0: 'Daylight', 1: 'Darkness - lights lit',
                                                                2: 'Darkness - lights unlit / no lighting'}},
    'weather_conditions': {'name': 'Weather conditions',
                           'values': {0: 'Fine', 1: 'Raining/Snowing no high winds', 2: 'Raining/Snowing + high winds',
                                      3: 'Fog or mist'}},
    'road_surface_conditions': {'name': 'Road surface conditions',
                                'values': {0: 'Dry', 1: 'Wet or damp / Mud', 2: 'Snow / Frost or ice',
                                           3: 'Flood over 3cm deep / Oil or diesel'}},
    'has_special_condition': {'name': 'Special conditions at site',
                              'values': {0: 'None / unknown', 1: 'Special conditions at site'}},
    'has_road_hazard': {'name': 'Road hazard', 'values': {0: 'None', 1: 'Hazard in the road'}},
    'is_rural': {'name': 'Area type', 'values': {0: 'Urban/Unallocated area', 1: 'Rural'}},
    'vehicle_priority': {'name': 'Dominant vehicle type', 'values': {1: 'Private car/Taxi/Van', 2: 'Heavy vehicle/Bus',
                                                                     3: 'Two-wheeler/Horse/Mobility scooter'}},
    'towing_risk_group': {'name': 'Towing risk',
                          'values': {0: 'None', 1: 'Single trailer/Caravan', 2: 'Articulated/Double trailer'}},
    'lane_risk_group': {'name': 'Lane risk',
                        'values': {0: 'Main lane + MAX', 1: 'Shoulder', 2: 'Public transport/Electric lane',
                                   3: 'Sidewalk/Bicycle lane'}},
    'casualty_priority': {'name': 'Casualty priority',
                          'values': {1: 'Private car passenger', 2: 'Heavy vehicle passenger',
                                     3: 'Two-wheeler/Scooter rider', 4: 'Pedestrian'}},
    'did_skid': {'name': 'Skidding', 'values': {0: 'None', 1: 'Skidded'}},
    'hit_previous_accident': {'name': 'Hit previous accident', 'values': {0: 'None', 1: 'Previous accident'}},
    'vehicle_leaving_carriageway': {'name': 'Leaving carriageway',
                                    'values': {0: 'Did not leave carriageway', 1: 'Nearside / Straight ahead',
                                               2: 'Rebounded / Offside to central reservation',
                                               3: 'Crossed central reservation'}},
    'hit_object_off_carriageway': {'name': 'Hit object off carriageway',
                                   'values': {0: 'None', 1: 'Sign/Pole/Bus stop/Permanent object',
                                              2: 'Crash barrier / Ditch / Wall or fence',
                                              3: 'Tree / Central barrier / Water'}},
    'age_of_vehicle': {'name': 'Age of vehicle (Years)', 'values': 'numeric'},
    'escooter_flag': {'name': 'E-scooter involvement',
                      'values': {0: 'Vehicle was not an e-scooter', 1: 'Vehicle was an e-scooter'}},
    'casualty_class': {'name': 'Casualty class', 'values': {1: 'Driver or rider', 2: 'Passenger', 3: 'Pedestrian'}},
    'has_young_driver': {'name': 'Young driver (<25)', 'values': {0: 'No', 1: 'Yes'}},
    'has_elderly_driver': {'name': 'Elderly driver (>75)', 'values': {0: 'No', 1: 'Yes'}},
    'has_young_casualty': {'name': 'Young casualty (<20)', 'values': {0: 'No', 1: 'Yes'}},
    'has_elderly_casualty': {'name': 'Elderly casualty (>75)', 'values': {0: 'No', 1: 'Yes'}}
}


def predict_accident_with_explanation_cascade(input_data_row):
    """
    פונקציית הסבר מותאמת למודל הדו-שלבי (Cascade).
    מחליטה איזה מודל סיפק את ההחלטה הסופית וגוזרת ממנו את ערכי ה-SHAP בצורה בטוחה.
    """
    # 1. בדיקת מודל שלב 1
    probs_stage1 = model_stage1.predict_proba(input_data_row)[0]
    pred_stage1 = 0 if probs_stage1[0] > 0.5 else 1  # 0=Slight, 1=Major

    if pred_stage1 == 0:
        diagnosis = "Slight"
        confidence = probs_stage1[0] * 100
        deciding_model = model_stage1
        print("\nComputing SHAP explanation from Stage 1 model...")
    else:
        probs_stage2 = model_stage2.predict_proba(input_data_row)[0]
        pred_stage2 = 0 if probs_stage2[0] > 0.5 else 1  # 0=Serious, 1=Fatal

        if pred_stage2 == 0:
            diagnosis = "Serious"
            confidence = (probs_stage1[1] * probs_stage2[0]) * 100
        else:
            diagnosis = "Fatal"
            confidence = (probs_stage1[1] * probs_stage2[1]) * 100

        deciding_model = model_stage2
        print("\nComputing SHAP explanation from Stage 2 model...")

    # 2. חישוב SHAP על המודל שהכריע
    explainer = shap.TreeExplainer(deciding_model)
    shap_values = explainer.shap_values(input_data_row)

    # חילוץ בטוח של הערכים מ-SHAP כדי למנוע שגיאות Dimensions ואזהרות
    if isinstance(shap_values, list):
        # LightGBM לרוב מחזיר רשימה של 2 מערכים (אחד לכל מחלקה) בסיווג בינארי
        current_shap_values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif len(shap_values.shape) == 3:
        current_shap_values = shap_values[0, :, 1]
    else:
        current_shap_values = shap_values[0] if len(shap_values.shape) == 2 else shap_values

    # שימוש בערך מוחלט כדי למצוא את ההשפעה החזקה ביותר (לטובה או לרעה)
    feature_contributions = pd.DataFrame({
        'feature': X.columns,
        'impact_magnitude': np.abs(current_shap_values)
    }).sort_values(by='impact_magnitude', ascending=False)

    top_features = feature_contributions.head(2)['feature'].values

    # 3. תרגום הפיצ'רים (אנגלית)
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

    # 4. הפלט הסופי
    strength = f"The accident was classified as '{diagnosis}' with {confidence:.1f}% confidence."
    explanation = f"The prediction was primarily influenced by the following factors: [{descriptions[0]}] and [{descriptions[1]}]."

    return diagnosis, strength, explanation


# ==========================================
# 8. הרצת דוגמה חיה (Inference Test)
# ==========================================

sample_incident = X_test.iloc[[0]]

diag, strength, expl = predict_accident_with_explanation_cascade(sample_incident)

print("\n" + "X" * 70)
print("--- Triple Output for Sample Case (Cascade) ---")
print(f"1. Diagnosis: {diag}")
print(f"2. Confidence: {strength}")
print(f"3. Explanation: {expl}")
print("X" * 70)

import requests


# ==========================================
# 9. סימולציית חיבור API של מזג אוויר בזמן אמת (מבוסס שם עיר)
# ==========================================

def get_live_weather_and_map_to_model(city_name, api_key):
    """
    מתחבר ל-API של מזג האוויר לפי שם עיר ומתרגם את התוצאה לקידוד של המודל
    """
    print(f"\nFetching live weather data for city: {city_name}...")

    # חיבור ל-OpenWeatherMap דרך שם העיר (q=city_name)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)

        # וידוא שהבקשה עברה בהצלחה (קוד 200)
        if response.status_code != 200:
            print(f"API Error ({response.status_code}): {response.json().get('message')}")
            print("Using safe defaults (Clear, Dry).")
            return {'weather_conditions': 0, 'road_surface_conditions': 0}

        data = response.json()
        weather_main = data['weather'][0]['main'].lower()  # למשל: 'rain', 'clear', 'clouds'
        wind_speed = data['wind']['speed']  # מהירות רוח במטר לשנייה

        # --- תרגום מה-API לקידוד של המודל ---

        # 1. מזג אוויר (0=Fine, 1=Rain/Snow, 2=Rain/Snow+Wind, 3=Fog)
        weather_code = 0
        if 'rain' in weather_main or 'snow' in weather_main:
            weather_code = 2 if wind_speed > 10 else 1  # רוח חזקה מעל 10 מ/ש משנה קידוד
        elif 'fog' in weather_main or 'mist' in weather_main:
            weather_code = 3

        # 2. מצב כביש (0=Dry, 1=Wet, 2=Snow/Ice)
        road_code = 0
        if 'rain' in weather_main:
            road_code = 1
        elif 'snow' in weather_main:
            road_code = 2

        print(
            f"API Weather: {weather_main.capitalize()} (Wind: {wind_speed}m/s) -> Mapped Weather Code: {weather_code}, Road Code: {road_code}")

        return {
            'weather_conditions': weather_code,
            'road_surface_conditions': road_code
        }

    except Exception as e:
        print(f"API Fetch Failed: {e}. Using safe defaults (Clear, Dry).")
        return {'weather_conditions': 0, 'road_surface_conditions': 0}


# ==========================================
# 10. הדגמת הזרימה השלמה (ממשק משתמש -> API -> מודל)
# ==========================================

print("\n" + "=" * 50)
print("--- END-TO-END SIMULATION: UI + API + MODEL ---")
print("=" * 50)

# שלב א': ה-UI שולח לנו את הנתונים שהמשתמש הזין.
# נשתמש בשורה מהטסט כבסיס
ui_input_df = X_test.iloc[[0]].copy()

# העיר שהמשתמש הקליד באפליקציה (במקום GPS)
user_city = "Tel Aviv,IL"

# מפתח ה-API האמיתי והפעיל שלך
YOUR_REAL_API_KEY = "b245a156ed259736300c7d7e57ec0e4e"

# שלב ב': ה-Backend שולף את מזג האוויר החי לפי העיר
live_weather_features = get_live_weather_and_map_to_model(user_city, YOUR_REAL_API_KEY)

# שלב ג': הזרקת נתוני מזג האוויר האמיתיים לתוך ה-DataFrame
for col, val in live_weather_features.items():
    if col in ui_input_df.columns:
        ui_input_df[col] = val

# שלב ד': הרצת המודל הדו-שלבי וה-SHAP על הנתונים המעודכנים
diag, strength, expl = predict_accident_with_explanation_cascade(ui_input_df)

print("\n" + "X" * 70)
print("--- Final Output Sent Back to UI ---")
print(f"1. Diagnosis: {diag}")
print(f"2. Confidence: {strength}")
print(f"3. Explanation: {expl}")
print("X" * 70)