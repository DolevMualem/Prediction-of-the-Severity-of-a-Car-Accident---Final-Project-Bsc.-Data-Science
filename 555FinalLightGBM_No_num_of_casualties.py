import pandas as pd
import numpy as np
import os
import warnings
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests

# השתקת אזהרות מיותרות (כולל SHAP) לקבלת פלט נקי
warnings.filterwarnings("ignore")

# ==========================================
# 1. טעינת הנתונים והכנתם
# ==========================================
print("Loading data...")
# הגדרת נתיב הקובץ הדינמי (שולחן העבודה)
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals_No_num_of_casualties.csv')

df = pd.read_csv(file_path)
target_col = 'collision_severity'

X = df.drop(columns=[target_col])

# קידוד למשתנה מטרה בינארי:
# 1 (Fatal) ו-2 (Serious) -> הופכים ל-1 (Major)
# 3 (Slight) -> הופך ל-0 (Slight)
y = np.where(df[target_col].isin([1, 2]), 1, 0)

# פיצול מבוקר לסט אימון (70%) ובדיקה (30%) תוך שמירה על פרופורציית המחלקות (Stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# 2. אימון המודל הסופי (Tuned LightGBM via Optuna)
# ==========================================
print("\n--- Training Final LightGBM Model ---")
# הערה: הפרמטרים להלן והמשקולות נמצאו לאחר כיוונון היפר-פרמטרים מקיף באמצעות ספריית Optuna.
# נבחר הקנס 2.083 למחלקה 1 כדי למקסם Recall ללא איבוד קיצוני של הדיוק.
OPTUNA_PENALTY = {0: 1.0, 1: 2.083}

model_binary = lgb.LGBMClassifier(
    class_weight=OPTUNA_PENALTY,
    n_estimators=500,
    learning_rate=0.044,
    num_leaves=101,
    max_depth=7,
    min_child_samples=92,
    subsample=0.784,
    colsample_bytree=0.784,
    random_state=42,
    n_jobs=-1,
    verbose=-1  # <--- השורה הזו משתיקה את כל ה-Logs והאזהרות שקיבלת!
)

# אימון המודל על דאטה האימון
model_binary.fit(X_train, y_train)
print("--- Training Complete ---")

# ==========================================
# 3. ביצוע חיזוי (Inference) והערכת ביצועים
# ==========================================
print("\nPerforming Inference on Test Set...")
final_predictions = model_binary.predict(X_test)

# חישוב ביצועי Recall לכל מחלקה וסיכומם
recalls = recall_score(y_test, final_predictions, average=None)
custom_recall_score = recalls.sum()

print("\n" + "=" * 50)
print("--- Final Binary Optuna Tuned Model Results ---")
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
importances = model_binary.feature_importances_
importances_pct = (importances / importances.sum()) * 100

# סידור ובחירת 5 המאפיינים המשפיעים ביותר במודל
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Pct': importances_pct
}).sort_values(by='Importance_Pct', ascending=False).head(5)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
ax = sns.barplot(x='Importance_Pct', y='Feature', data=fi_df, hue='Feature', palette="viridis", legend=False)

# הוספת הערכים באחוזים על גבי כל עמודה בגרף
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{width:.1f}%', (width, p.get_y() + p.get_height() / 2),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points',
                fontsize=11, fontweight='bold', color='black')

plt.title('Top 5 Influential Features - Optuna LightGBM\n(Normalized to 100%)', fontsize=16, fontweight='bold')
plt.xlabel('Contribution to Model Decision (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, fi_df['Importance_Pct'].max() + 5)
plt.tight_layout()
plt.show()

# ==========================================
# 5. יישום פלט משולש (Triple Output) למודל בינארי + מילון
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

def predict_accident_binary_lgb(input_data_row):
    """
    מבצעת חיזוי על שורה בודדת (מקרה תאונה אחד) ומספקת הסבר קריא באמצעות SHAP.
    מותאם למודל הבינארי היחיד (LightGBM).
    """
    probs = model_binary.predict_proba(input_data_row)[0]
    pred_class = int(model_binary.predict(input_data_row)[0])

    class_names = {0: "Slight", 1: "Major (Serious/Fatal)"}
    diagnosis = class_names[pred_class]
    confidence = probs[pred_class] * 100

    print(f"\nComputing SHAP explanation for class: {diagnosis}...")

    explainer = shap.TreeExplainer(model_binary)
    shap_values = explainer.shap_values(input_data_row)

    # חילוץ חכם של ערכי ה-SHAP בהתאם לפלט האפשרי של LightGBM (רשימה או מערך)
    current_shap_values = shap_values[1][0] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values[0] if isinstance(shap_values, list) else shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

    feature_contributions = pd.DataFrame({
        'feature': X.columns,
        'impact_magnitude': np.abs(current_shap_values)
    }).sort_values(by='impact_magnitude', ascending=False)

    top_features = feature_contributions.head(2)['feature'].values

    # תרגום המאפיינים המשפיעים ביותר לשפה קריאה
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
# 6. חיבור API של מזג אוויר וסימולציית UI
# ==========================================
def get_live_weather_and_map_to_model(city_name, api_key):
    """
    מושכת נתוני מזג אוויר חיים מה-API וממירה אותם לקודים שהמודל מכיר.
    """
    print(f"\nFetching live weather data for city: {city_name}...")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {'weather_conditions': 0, 'road_surface_conditions': 0}

        data = response.json()
        weather_main = data['weather'][0]['main'].lower()
        wind_speed = data['wind']['speed']

        # מיפוי מזג האוויר והכביש בהתאם לקידוד המקורי בדאטה-סט
        weather_code = 2 if ('rain' in weather_main or 'snow' in weather_main) and wind_speed > 10 else 1 if ('rain' in weather_main or 'snow' in weather_main) else 3 if ('fog' in weather_main or 'mist' in weather_main) else 0
        road_code = 1 if 'rain' in weather_main else 2 if 'snow' in weather_main else 0

        print(f"API Weather: {weather_main.capitalize()} (Wind: {wind_speed}m/s) -> Mapped Weather Code: {weather_code}, Road Code: {road_code}")
        return {'weather_conditions': weather_code, 'road_surface_conditions': road_code}

    except Exception:
        return {'weather_conditions': 0, 'road_surface_conditions': 0}


# === הרצת הסימולציה מול שורת מבחן ===
print("\n" + "=" * 50)
print("--- END-TO-END SIMULATION: UI + API + MODEL ---")
print("=" * 50)

# לקיחת מקרה בודד מתוך סט הבדיקה שידמה הזנת משתמש בממשק
ui_input_df = X_test.iloc[[0]].copy()
user_city = "Tel Aviv,IL"
YOUR_REAL_API_KEY = "b245a156ed259736300c7d7e57ec0e4e"

# שליפה ועדכון של נתוני מזג האוויר החיים
live_weather_features = get_live_weather_and_map_to_model(user_city, YOUR_REAL_API_KEY)
for col, val in live_weather_features.items():
    if col in ui_input_df.columns:
        ui_input_df[col] = val

# חילוץ התחזית וההסבר של המקרה הספציפי
diag, strength, expl = predict_accident_binary_lgb(ui_input_df)

print("\n" + "X" * 70)
print("--- Final Output Sent Back to UI ---")
print(f"1. Diagnosis: {diag}")
print(f"2. Confidence: {strength}")
print(f"3. Explanation: {expl}")
print("X" * 70)

# ==========================================
# [אימות נתונים] בדיקה והדפסת התפלגות הדאטה
# ==========================================
print("\n" + "="*40)
print("     DATASET DISTRIBUTION VERIFICATION     ")
print("="*40)
total_rows = len(df)
slight_count = np.sum(y == 0)
major_count = np.sum(y == 1)
actual_ratio = slight_count / major_count if major_count > 0 else 0

print(f"Total Dataset Rows:       {total_rows:,}")
print(f"Slight Severity (0):      {slight_count:,}")
print(f"Major Severity (1):       {major_count:,}")
print(f"Operational Data Ratio:   1 : {actual_ratio:.2f}")
print("="*40 + "\n")

# ===========================================================================================
# 4. הפקת לוח מחוונים ויזואלי לפוסטר הגמר
# ==========================================
print("\nGenerating advanced visualization dashboard for final poster...")

import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

# הגדרת מבנה הקנבס המאוחד (רוחב 20, גובה 12)
fig = plt.figure(figsize=(20, 12), facecolor='#f8f9fa')
# יצירת גריד של 3 שורות: שורת KPI, ושתי שורות לגרפים המרכזיים
gs = gridspec.GridSpec(3, 2, height_ratios=[0.25, 0.45, 0.45], hspace=0.4, wspace=0.3)

# ------------------------------------------
# א' - יצירת בועות / קוביות מדדי דאטה (KPI Blocks)
# ------------------------------------------
kpi_data = [
    {"title": "Features Reduction", "val": "99 ──> 26", "color": "#2c3e50"},
    {"title": "Initial Fatal Cases", "val": "~1.5%", "color": "#c0392b"},
    {"title": "Added Fatal Cases", "val": "97,783", "color": "#27ae60"},
    {"title": "Total Collision (Rows)", "val": "601,261", "color": "#2980b9"},
    {"title": "Major:Slight Ratio", "val": "1 : 1.8", "color": "#8e44ad"}
]

# חלוקה דינמית של 5 התיבות לאורך השורה הראשונה
kpi_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :])

for i, kpi in enumerate(kpi_data):
    ax_kpi = fig.add_subplot(kpi_gs[0, i])
    ax_kpi.set_facecolor('white')

    # עיצוב מסגרת מעוגלת/נקייה לתיבה
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor('#e2e8f0')
        spine.set_linewidth(1.5)

    # כתיבת הערך המספרי המרכזי (גדול ובולט)
    ax_kpi.text(0.5, 0.55, kpi["val"], fontsize=22, fontweight='bold',
                ha='center', va='center', color=kpi["color"])
    # כתיבת כותרת המדד
    ax_kpi.text(0.5, 0.15, kpi["title"], fontsize=11, fontweight='semibold',
                ha='center', va='center', color='#64748b')

    ax_kpi.set_xticks([])
    ax_kpi.set_yticks([])

# ------------------------------------------
# ב' - [Part 4] גרף חשיבות משתנים מותאם (Top 5 Features)
# ------------------------------------------
ax_fi = fig.add_subplot(gs[1:, 0])

# חישוב ונרמול חשיבות הפיצ'רים ל-100%
importances = model_binary.feature_importances_
importances_pct = (importances / importances.sum()) * 100

fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Pct': importances_pct
}).sort_values(by='Importance_Pct', ascending=False).head(5)  # בחירת 5 המובילים בלבד

# הפקת גרף עמודות אופקי מעוצב
sns.set_style("whitegrid")
ax_bar = sns.barplot(x='Importance_Pct', y='Feature', data=fi_df,
                     hue='Feature', palette="viridis", legend=False, ax=ax_fi)

# הוספת תוויות אחוזים דינמיות על גבי העמודות
for p in ax_bar.patches:
    width = p.get_width()
    ax_bar.annotate(f'{width:.1f}%', (width, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', xytext=(7, 0), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='#1e293b')

ax_fi.set_title('Top 5 Most Influential Features', fontsize=15, fontweight='bold',
                pad=15, color='#1e293b')
ax_fi.set_xlabel('Contribution to Model Decision (%)', fontsize=12, fontweight='semibold', color='#475569')
ax_fi.set_ylabel('Feature Name', fontsize=12, fontweight='semibold', color='#475569')
ax_fi.set_xlim(0, fi_df['Importance_Pct'].max() + 8)
ax_fi.tick_params(axis='both', labelsize=11)

# ------------------------------------------
# ג' - [Part 5] מפת חום של מטריצת הבלבול (Confusion Matrix Heatmap)
# ------------------------------------------
ax_cm = fig.add_subplot(gs[1:, 1])

# חישוב מטריצת הבלבול האמיתית מתוך סט הטסט
cm = confusion_matrix(y_test, final_predictions)

# בניית מפת חום ריבועית צבעונית עם כותרות אקדמיות
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', cbar=False,
            square=True, annot_kws={"size": 16, "weight": "bold"},
            xticklabels=['Predicted Slight (0)', 'Predicted Major (1)'],
            yticklabels=['Actual Slight (0)', 'Actual Major (1)'], ax=ax_cm)

# התאמת עיצוב הכותרות והצירים לרמה אקדמית באנגלית
ax_cm.set_title('Final Confusion Matrix Evaluation', fontsize=15, fontweight='bold',
                pad=15, color='#1e293b')
ax_cm.set_xlabel('Predicted Label', fontsize=13, fontweight='semibold', labelpad=10, color='#475569')
ax_cm.set_ylabel('Actual Label', fontsize=13, fontweight='semibold', labelpad=10, color='#475569')
ax_cm.tick_params(axis='both', labelsize=12, rotation=0)

# הוספת כותרת על-לכל הלוח (Super Title)
plt.suptitle("Model Evaluation & Data Pipeline Metrics Dashboard", fontsize=20, fontweight='bold', y=0.96,
             color='#0f172a')

output_image_path = os.path.join(desktop_path, 'final_model_dashboard.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
# הצגת הלוח המאוחד
plt.show()

# ==========================================
# [קוד מתוקן - ללא שגיאות] ויזואליזציה של ה-Classification Report
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\nGenerating high-contrast visual Classification Report...")

# 1. בניית ה-DataFrame על בסיס נתוני האמת המדויקים של המודל
report_data = {
    'Precision': [0.82, 0.60],
    'Recall': [0.74, 0.72],
    'F1-Score': [0.78, 0.66]
}
report_df = pd.DataFrame(report_data, index=['Slight (Class 0)', 'Major (Class 1)'])

# 2. הגדרת מימדי הקנבס ועיצוב הרקע
fig, ax = plt.subplots(figsize=(11, 7), facecolor='#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

# 3. הפקת מפת חום עם פלטת YlGnBu (צהוב-ירוק-כחול) לניגודיות מקסימלית
# הגדרת vmin=0.55 גורמת לכך שגם הערך הנמוך ביותר (0.60) יקבל גוון צבעוני עמוק ולא ייבלע ברקע
sns.heatmap(report_df, annot=True, fmt='.2f', cmap='YlGnBu', cbar=False,
            square=False, annot_kws={"size": 17, "weight": "bold"},
            linewidths=4, linecolor='#f8f9fa', vmin=0.55, vmax=0.85, ax=ax)

# 4. הגדרת כותרות וצירים באנגלית ברמה אקדמית
plt.title('Model Classification Report Metrics',
          fontsize=14, fontweight='bold', pad=25, color='#1e293b')

# הגדרת הגופן והדגש לצירי המטריצה
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='semibold', color='#475569')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='semibold', color='#475569', rotation=0)

# הזזת תוויות ציר ה-X לחלק העליון כמקובל בטבלאות מדעי הנתונים
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# 5. הבלטה מודגשת ובולטת של ה-Accuracy (תיבה כחולה כהה במרכז) - תוקן ללא shadow
ax.text(0.5, -0.15, "Overall Model Accuracy: 73.0%",
        fontsize=15, fontweight='bold', color='white', ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", fc="#1e3a8a", ec="#1d4ed8", lw=1.5))

# 6. הצנעה מוצנעת של נתוני ה-Support (טקסט קטן, דק ואפור בתחתית הרחוקה)
ax.text(0.5, -0.26, "Tested on a stratified validation subset of 180,378 historical incident rows.",
        fontsize=10, fontweight='normal', color='#94a3b8', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()

# 7. שמירה אוטומטית של התמונה באיכות גבוהה (300 DPI) לשולחן העבודה
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
output_image_path = os.path.join(desktop_path, 'high_contrast_classification_report.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

print(f"Success! High-contrast plot saved to desktop as: {output_image_path}")

# הצגת הגרף על המסך
plt.show()

import joblib
joblib.dump(model_binary, 'no_casualties_lightgbm_model.pkl')

print("המודל הבינארי נשמר בהצלחה לפרויקט!")

