import pandas as pd
import os

# 1. הגדרת נתיב לקובץmerged_vehicle_casualty שנמצא על שולחן העבודה
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_path = os.path.join(desktop_path, 'merged_vehicle_casualty.csv')

# טעינת הנתונים
# הערה: אם הקובץ הוא אקסל, החלף ל-pd.read_excel
df = pd.read_csv(file_path)

# 2. רשימת העמודות להסרה כפי שביקשת
columns_to_drop = [
    'vehicle_direction_from',
    'vehicle_direction_to',
    'junction_location',
    'propulsion_code',
    'generic_make_model',
    'enhanced_casualty_severity',
    'casualty_injury_based',
    'sex_of_casualty',
    'casualty_adjusted_severity_serious',
    'casualty_adjusted_severity_slight',
    'collision_year_y'

]

# 3. הסרת העמודות
df_cleaned = df.drop(columns=columns_to_drop)

# הצגת מידע ראשוני לוודא שהעמודות הוסרו
print("העמודות הוסרו בהצלחה.")
print(f"מספר עמודות שנותרו: {len(df_cleaned.columns)}")
print("\nרשימת העמודות שנותרו בטבלה:")
print(df_cleaned.columns.tolist())

# 4. שמירת הקובץ החדש לשולחן העבודה (אופציונלי)
output_path = os.path.join(desktop_path, 'merged_vehicle_casualty_clean2.csv')
df_cleaned.to_csv(output_path, index=False)
print(f"\nmerged_vehicle_casualty_clean.csv")