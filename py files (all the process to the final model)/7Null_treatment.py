import pandas as pd
import numpy as np
import os

# 1. הגדרת נתיבים וטעינת הנתונים (מבוסס על הקוד הקודם שלך)
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_path = os.path.join(desktop_path, 'Merged_Table_32_Collomns.csv')

print("⏳ טוען נתונים לביצוע ניקוי סופי...")
df = pd.read_csv(file_path, low_memory=False)

# --- שלב א': הסרת עמודות עם אחוז חסרים גבוה (מעל 30%) ---
cols_to_drop = ['junction_control', 'first_road_class']
df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"✅ עמודות {cols_to_drop} הוסרו בהצלחה.")

# --- שלב ב': הפיכת special_conditions_at_site לבינארית ---
# לוגיקה: 0 (None), -1 (Missing), 9 (Unknown) הופכים ל-0
# הערכים 1, 2, 3, 4, 5, 6, 7 (מפגעים) הופכים ל-1
def transform_special_conditions(val):
    if val in [1, 2, 3, 4, 5, 6, 7]:
        return 1
    return 0

df['has_special_condition'] = df['special_conditions_at_site'].apply(transform_special_conditions)
df = df.drop(columns=['special_conditions_at_site'])
print("✅ עמודת special_conditions_at_site עובדה לבינארית (has_special_condition).")

# --- שלב ג': מילוי ערכים שכיחים (Mode Imputation) ל-11 עמודות ---
# הגדרת המיפוי: לכל עמודה ציינו אילו קודים נחשבים "חוסר מידע" ומהו הערך השכיח שבדקנו
mode_imputation_config = {
    'vehicle_leaving_carriageway': {'null_codes': [9, -1], 'mode_val': 0},
    'hit_object_off_carriageway': {'null_codes': [99, -1], 'mode_val': 0},
    'pedestrian_crossing': {'null_codes': [99, -1], 'mode_val': 0},
    'junction_detail': {'null_codes': [99, -1], 'mode_val': 0},
    'weather_conditions': {'null_codes': [9, -1], 'mode_val': 1},
    'carriageway_hazards': {'null_codes': [99, -1], 'mode_val': 0},
    'road_type': {'null_codes': [9, -1], 'mode_val': 6},
    'road_surface_conditions': {'null_codes': [9, -1], 'mode_val': 1},
    'speed_limit': {'null_codes': [99, -1], 'mode_val': 30},
    'light_conditions': {'null_codes': [-1], 'mode_val': 1},
    'urban_or_rural_area': {'null_codes': [-1], 'mode_val': 1}
}

print("🩹 מבצע מילוי ערכים שכיחים ל-11 עמודות...")
for col, config in mode_imputation_config.items():
    if col in df.columns:
        # החלפת הקודים הבעייתיים וגם ערכי NaN (אם קיימים) בערך השכיח
        df[col] = df[col].replace(config['null_codes'], config['mode_val'])
        df[col] = df[col].fillna(config['mode_val'])

print("✅ מילוי השכיחים הסתיים.")

# --- שלב ד': וידוא סופי ושמירה ---
# בדיקה שאין יותר ערכים חסרים בעמודות שטיפלנו בהן
print("\n🔍 וידוא סופי של ערכים חסרים בעמודות המטופלות:")
for col in mode_imputation_config.keys():
    if col in df.columns:
        null_check = df[col].isin(mode_imputation_config[col]['null_codes']).sum()
        print(f"   - {col}: {null_check} ערכים לא תקינים נותרו.")

# שמירה לקובץ חדש
output_final = os.path.join(desktop_path, 'Final_Dataset_Cleaned_Ready.csv')
df.to_csv(output_final, index=False)

print("-" * 50)
print(f"🏁 תהליך הניקוי הושלם בהצלחה!")
print(f"💾 הקובץ הסופי נשמר בשם: Final_Dataset_Cleaned_Ready.csv")