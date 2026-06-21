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
y = df[target_col] - 1  # 0=Fatal, 1=Serious, 2=Slight

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
# 2. אימון מודל ה-XGBoost (עם פרמטרי הזהב של Optuna)
# ==========================================
print("\nTraining Final XGBoost Model with Optuna parameters...")

xgb_params = {
    'objective': 'multi:softprob',  # מאפשר לקבל אחוזי ביטחון מדויקים
    'num_class': 3,
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
print("--- Final XGBoost Results ---")
print("=" * 50)
print(f"Overall Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
print(f"⭐ Sum of Recalls: {custom_recall_score:.2f} ⭐\n")

print("Classification Report (0=Fatal, 1=Serious, 2=Slight):")
print(classification_report(y_test, final_predictions))

print("\nFinal Confusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
cm_df = pd.DataFrame(cm, index=['Actual Fatal (0)', 'Actual Serious (1)', 'Actual Slight (2)'],
                     columns=['Pred Fatal (0)', 'Pred Serious (1)', 'Pred Slight (2)'])
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

plt.title('Top 10 Influential Features - Final XGBoost\n(Normalized to 100%)', fontsize=16, fontweight='bold')
plt.xlabel('Contribution to Model Decision (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, fi_df['Importance_Pct'].max() + 5)
plt.tight_layout()
plt.show()

# ==========================================
# 5. מילון תרגום לממשק (UI Translation Dictionary)
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
    'escooter_flag': {'name': 'E-scooter involvement',
                      'values': {0: 'Vehicle was not an e-scooter', 1: 'Vehicle was an e-scooter'}},
    'casualty_class': {'name': 'Casualty class', 'values': {1: 'Driver or rider', 2: 'Passenger', 3: 'Pedestrian'}},
    'has_young_driver': {'name': 'Young driver (<25)', 'values': {0: 'No', 1: 'Yes'}},
    'has_elderly_driver': {'name': 'Elderly driver (>75)', 'values': {0: 'No', 1: 'Yes'}},
    'has_young_casualty': {'name': 'Young casualty (<20)', 'values': {0: 'No', 1: 'Yes'}},
    'has_elderly_casualty': {'name': 'Elderly casualty (>75)', 'values': {0: 'No', 1: 'Yes'}}
}

# ==========================================
# הדפסת 10 העמודות המובילות בחשיבותן
# ==========================================
print("\n" + "=" * 50)
print("Top 10 Influential Features (Feature Importance):")
print("=" * 50)

# חילוץ החשיבות מהמודל
importances = final_xgb_model.feature_importances_
feature_names = X.columns

# יצירת DataFrame לסידור הנתונים
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)

# הדפסה בפורמט טבלאי נוח
for index, row in fi_df.iterrows():
    print(f"{row['Feature']:<30} | {row['Importance']:.4f}")
print("=" * 50)

# ==========================================
# 6. פונקציית פלט משולש למודל XGBoost (Diagnosis, Confidence, Explanation)
# ==========================================
def predict_accident_with_explanation(input_data_row):
    """
    מבצעת ניבוי על שורה בודדת מול מודל ה-XGBoost, מחשבת ביטחון (Confidence)
    ומייצרת הסבר אנושי באנגלית בעזרת SHAP.
    """
    # 1. חיזוי ובדיקת הסתברויות
    probs = final_xgb_model.predict_proba(input_data_row)[0]
    pred_class = int(final_xgb_model.predict(input_data_row)[0])

    class_names = {0: "Fatal", 1: "Serious", 2: "Slight"}
    diagnosis = class_names[pred_class]
    confidence = probs[pred_class] * 100

    print(f"\nComputing SHAP explanation for class: {diagnosis}...")

    # 2. חישוב SHAP
    explainer = shap.TreeExplainer(final_xgb_model)
    shap_values = explainer.shap_values(input_data_row)

    # חילוץ הערכים עבור המחלקה שנחזתה (טיפול בפורמטים השונים של XGBoost)
    if isinstance(shap_values, list):
        current_shap_values = shap_values[pred_class][0]
    elif len(shap_values.shape) == 3:
        current_shap_values = shap_values[0, :, pred_class]
    else:
        current_shap_values = shap_values[0]

    # מציאת 2 המאפיינים שהשפיעו הכי הרבה
    feature_contributions = pd.DataFrame({
        'feature': X.columns,
        'impact_magnitude': np.abs(current_shap_values)
    }).sort_values(by='impact_magnitude', ascending=False)

    top_features = feature_contributions.head(2)['feature'].values

    # 3. תרגום לאנגלית קריאה מהמילון
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

    # 4. יצירת הפלט הסופי
    strength = f"The accident was classified as '{diagnosis}' with {confidence:.1f}% confidence."
    explanation = f"The prediction was primarily influenced by the following factors: [{descriptions[0]}] and [{descriptions[1]}]."

    return diagnosis, strength, explanation


# ==========================================
# 7. חיבור API של מזג אוויר
# ==========================================
def get_live_weather_and_map_to_model(city_name, api_key):
    print(f"\nFetching live weather data for city: {city_name}...")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"API Error ({response.status_code}): {response.json().get('message')}")
            return {'weather_conditions': 0, 'road_surface_conditions': 0}

        data = response.json()
        weather_main = data['weather'][0]['main'].lower()
        wind_speed = data['wind']['speed']

        weather_code = 0
        if 'rain' in weather_main or 'snow' in weather_main:
            weather_code = 2 if wind_speed > 10 else 1
        elif 'fog' in weather_main or 'mist' in weather_main:
            weather_code = 3

        road_code = 0
        if 'rain' in weather_main:
            road_code = 1
        elif 'snow' in weather_main:
            road_code = 2

        print(
            f"API Weather: {weather_main.capitalize()} (Wind: {wind_speed}m/s) -> Mapped Weather Code: {weather_code}, Road Code: {road_code}")
        return {'weather_conditions': weather_code, 'road_surface_conditions': road_code}

    except Exception as e:
        print(f"API Fetch Failed: {e}. Using safe defaults (Clear, Dry).")
        return {'weather_conditions': 0, 'road_surface_conditions': 0}


# ==========================================
# 8. הדגמת הזרימה השלמה (UI -> API -> MODEL -> UI)
# ==========================================
print("\n" + "=" * 50)
print("--- END-TO-END SIMULATION: UI + API + MODEL ---")
print("=" * 50)

# נתונים שמגיעים מהאפליקציה של המוקדן
ui_input_df = X_test.iloc[[0]].copy()
user_city = "Tel Aviv,IL"
YOUR_REAL_API_KEY = "b245a156ed259736300c7d7e57ec0e4e"

# שליפת נתונים חיים
live_weather_features = get_live_weather_and_map_to_model(user_city, YOUR_REAL_API_KEY)

# הזרקת נתוני מזג האוויר האמיתיים לנתוני המוקדן
for col, val in live_weather_features.items():
    if col in ui_input_df.columns:
        ui_input_df[col] = val

# הרצת המודל ויצירת הסבר
diag, strength, expl = predict_accident_with_explanation(ui_input_df)

print("\n" + "X" * 70)
print("--- Final Output Sent Back to UI ---")
print(f"1. Diagnosis: {diag}")
print(f"2. Confidence: {strength}")
print(f"3. Explanation: {expl}")
print("X" * 70)