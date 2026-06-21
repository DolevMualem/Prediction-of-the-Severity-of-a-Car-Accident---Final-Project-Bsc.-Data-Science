import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 1. טעינת הנתונים
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model.csv')
df = pd.read_csv(file_path)

# 2. הכנה זמנית ל-XGBoost (חייב להתחיל מ-0)
# 1 -> 0 (Fatal), 2 -> 1 (Serious), 3 -> 2 (Slight)
target_col = 'collision_severity'
X = df.drop(columns=[target_col])
y = df[target_col] - 1

# 3. פיצול ל-70% אימון ו-30% בדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 4. הגדרת משקולות ענישה מותאמות
# הורדנו ל-35 כדי לשפר Precision ודיוק כללי (Accuracy)
custom_weights_map = {0: 30.0, 1: 3.0, 2: 1.0}
sample_weights = y_train.map(custom_weights_map)

# 5. הגדרת מודל XGBoost עם הפרמטרים החדשים
print("Starting fine-tuned XGBoost model training on the full dataset...")
model_xgb = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=3,
    random_state=42,
    n_jobs=-1
)

# אימון
model_xgb.fit(X_train, y_train, sample_weight=sample_weights)

# 6. חיזוי והערכה (החזרת התוצאות לפורמט המקורי 1,2,3)
y_pred = model_xgb.predict(X_test)

print("\n" + "=" * 40)
print("--- Training Complete: XGBoost Optimized Results ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("=" * 40)

# 7. הצגת דוח סיווג (Labels 1,2,3)
print("\nClassification Report:")
print(classification_report(y_test + 1, y_pred + 1))

# 8. הצגת מטריצת בלבול בפורמט טבלה קריא
print("\nFinal Confusion Matrix:")
cm = confusion_matrix(y_test + 1, y_pred + 1)
cm_df = pd.DataFrame(cm, index=['Actual Fatal(1)', 'Actual Serious(2)', 'Actual Slight(3)'],
                     columns=['Pred Fatal(1)', 'Pred Serious(2)', 'Pred Slight(3)'])
print(cm_df)

# 9. בונוס: הדפסת חשיבות המשתנים (עשרת הגדולים)
print("\nTop 10 Most Influential Features on the Model:")
importances = pd.Series(model_xgb.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))

# 10. חילוץ חשיבות המשתנים לגרף
feature_names = X.columns
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Pct': importances * 100  # הפיכה לאחוזים
})

fi_df = fi_df.sort_values(by='Importance_Pct', ascending=False).head(10)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# שימוש ב-hue כדי למנוע את האזהרה של Seaborn
ax = sns.barplot(
    x='Importance_Pct',
    y='Feature',
    data=fi_df,
    hue='Feature',
    palette="Blues_r",
    legend=False
)

for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{width:.1f}%',
                (width, p.get_y() + p.get_height() / 2),
                ha='left', va='center',
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=11, fontweight='bold', color='black')

plt.title('Top 10 Most Influential Features\n(Normalized to 100% Total Contribution)', fontsize=16, fontweight='bold')
plt.xlabel('Contribution to Model Decision (%)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.xlim(0, fi_df['Importance_Pct'].max() + 5)
plt.tight_layout()
plt.show()

print(f"Total importance sum of ALL features: {sum(importances) * 100:.1f}%")

############# יישום פלט משולש + מילון מתורגם לאנגלית #########

# מילון תרגום מעמיק המבוסס על אפיון הפרויקט (מתוך התמונות) - הומר לאנגלית
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


def predict_accident_with_explanation(input_data_row):
    """
    מבצע חיזוי לתאונה, מחשב הסבר SHAP ומחזיר טקסט קריא וברור למשתמש (באנגלית)
    """
    severity_map = {0: "Fatal", 1: "Serious", 2: "Slight"}

    # 1. חיזוי קטגוריה וביטחון
    pred_idx = model_xgb.predict(input_data_row)[0]
    probs = model_xgb.predict_proba(input_data_row)[0]
    confidence = probs[pred_idx] * 100

    # 2. יצירת הסבר SHAP
    print("\nComputing SHAP explanation...")
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(input_data_row)

    if isinstance(shap_values, list):
        current_shap_values = shap_values[pred_idx].flatten()
    elif len(shap_values.shape) == 3:
        current_shap_values = shap_values[0, :, pred_idx]
    else:
        current_shap_values = shap_values.flatten()

    feature_contributions = pd.DataFrame({
        'feature': X.columns,
        'contribution': current_shap_values
    }).sort_values(by='contribution', ascending=False)

    top_features = feature_contributions.head(2)['feature'].values

    # 3. תרגום הפיצ'רים לשפה אנושית בעזרת המילון
    descriptions = []
    for f_name in top_features:
        actual_value = input_data_row[f_name].values[0]

        if f_name in detailed_translation:
            field_info = detailed_translation[f_name]
            field_name_eng = field_info['name']

            # בדיקה אם יש מיפוי קטגוריאלי לערך או שהוא מספרי חופשי
            if isinstance(field_info['values'], dict):
                val_eng = field_info['values'].get(actual_value, f"Code {actual_value}")
                descriptions.append(f"{field_name_eng}: {val_eng}")
            else:
                descriptions.append(f"{field_name_eng}: {actual_value}")
        else:
            descriptions.append(f"{f_name} ({actual_value})")

    # 4. בניית הפלט המשולש
    diagnosis = severity_map[pred_idx]
    strength = f"The accident was classified as '{diagnosis}' with {confidence:.1f}% confidence."
    explanation = f"The prediction was primarily influenced by the following factors: [{descriptions[0]}] and [{descriptions[1]}]."

    return diagnosis, strength, explanation


## דוגמא חיה
sample_incident = X_test.iloc[[0]]

diag, strength, expl = predict_accident_with_explanation(sample_incident)

print("\n" + "X" * 70)
print("--- Triple Output for Sample Case ---")
print(f"1. Diagnosis: {diag}")
print(f"2. Confidence: {strength}")
print(f"3. Explanation: {expl}")
print("X" * 70)