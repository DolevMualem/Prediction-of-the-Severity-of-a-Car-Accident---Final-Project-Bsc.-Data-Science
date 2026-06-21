import pandas as pd
import os

# 1. הגדרת נתיבים
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_path = os.path.join(desktop_path, 'Merged_Table_32_Collomns.csv')

if not os.path.exists(file_path):
    print(f"שגיאה: הקובץ {file_path} לא נמצא.")
else:
    # טעינת הנתונים
    df = pd.read_csv(file_path)

    print("-" * 50)
    print("📊 דוח ניתוח סטטיסטי - Final Dataset")
    print("-" * 50)

    # א. מימדי הקובץ
    print(f"🔹 מספר שורות: {len(df):,}")
    print(f"🔹 מספר עמודות: {len(df.columns)}")

    # ב. שמות כל העמודות
    print("\n🔹 רשימת כל העמודות בקובץ:")
    all_columns = df.columns.tolist()
    for i, col in enumerate(all_columns, 1):
        print(f"{i}. {col}")

    # ג. התפלגות עמודת המטרה (collision_severity)
    print("\n🔹 התפלגות עמודת המטרה (collision_severity):")
    severity_counts = df['collision_severity'].value_counts()
    severity_pct = df['collision_severity'].value_counts(normalize=True) * 100

    severity_map = {1: "Fatal (קטלנית)", 2: "Serious (קשה)", 3: "Slight (קלה)"}

    for sev, count in severity_counts.items():
        name = severity_map.get(sev, f"Unknown ({sev})")
        print(f"   - {name}: {count:,} תאונות ({severity_pct[sev]:.2f}%)")

    # ד. בדיקת ערכים חסרים (NULL)
    print("\n🔹 בדיקת ערכים חסרים (עמודות עם NULL בלבד):")
    null_data = df.isnull().sum()
    null_cols = null_data[null_data > 0]

    if null_cols.empty:
        print("   ✅ לא נמצאו ערכי NULL באף עמודה!")
    else:
        for col, count in null_cols.items():
            pct = (count / len(df)) * 100
            print(f"   - {col}: {count:,} חסרים ({pct:.2f}%)")

import pandas as pd
import numpy as np

# הגדרת מילון המיפוי של הערכים הריקים (Null-equivalent values) כפי שציינת
null_values_map = {
    'first_road_class': [6, -1],
    'road_type': [9, -1],
    'speed_limit': [99, -1],
    'junction_detail': [99, -1],
    'junction_control': [9, -1],
    'pedestrian_crossing': [99, -1],
    'light_conditions': [-1],
    'weather_conditions': [9, -1],
    'road_surface_conditions': [9, -1],
    'special_conditions_at_site': [9, -1],
    'carriageway_hazards': [99, -1],
    'urban_or_rural_area': [-1],
    'skidding_and_overturning': [9, -1],
    'hit_object_in_carriageway': [99, -1],
    'vehicle_leaving_carriageway': [9, -1],
    'hit_object_off_carriageway': [99, -1]
}


def analyze_missing_values(df, mapping):
    results = []

    for col, null_list in mapping.items():
        if col in df.columns:
            # ספירה כמה פעמים מופיעים הערכים הבעייתיים בעמודה
            missing_count = df[col].isin(null_list).sum()
            total_rows = len(df)
            percentage = (missing_count / total_rows) * 100

            results.append({
                'Column': col,
                'Missing_Count': missing_count,
                'Percentage (%)': round(percentage, 2),
                'Codes_Detected': null_list
            })
        else:
            print(f"אזהרה: העמודה {col} לא נמצאה בטבלה.")

    return pd.DataFrame(results)


# הרצת הניתוח על הטבלה שלך
missing_report = analyze_missing_values(df, null_values_map)

# הצגת הדוח
print("--- דוח ערכים חסרים (Unknown) לפי עמודות ---")
print(missing_report.sort_values(by='Percentage (%)', ascending=False))


# --- בדיקת קשר (Correlation) עבור special_conditions_at_site ---

def analyze_condition_impact(input_df):
    # יצירת עותק זמני לעבודה
    temp_df = input_df.copy()

    # 1. חלוקה ל-3 קבוצות לוגיות לצורך הבדיקה
    def categorize_condition(x):
        if x in [9, -1]: return 'Unknown'
        if x == 0: return 'None (Normal)'
        return 'Special Condition (Hazard)'  # עבודות בדרך, שמן וכו'

    temp_df['condition_group'] = temp_df['special_conditions_at_site'].apply(categorize_condition)

    # 2. חישוב התפלגות החומרה בכל קבוצה (באחוזים)
    # מזכיר: 1=Fatal, 2=Serious, 3=Slight
    severity_dist = pd.crosstab(temp_df['condition_group'], temp_df['collision_severity'], normalize='index') * 100

    # 3. הוספת עמודת ספירה כללית כדי לראות את גודל הקבוצה
    severity_dist['Total_Accidents'] = temp_df['condition_group'].value_counts()

    return severity_dist


# הרצת הניתוח
impact_report = analyze_condition_impact(df)

print("\n" + "=" * 60)
print("📊 ניתוח השפעת 'תנאים מיוחדים' על חומרת התאונה (באחוזים)")
print("=" * 60)
print(impact_report.round(2))

## רשימה מלאה של כל 11 העמודות שמתחת ל-10% חסר
all_low_null_cols = [
    'vehicle_leaving_carriageway',
    'hit_object_off_carriageway',
    'pedestrian_crossing',
    'junction_detail',
    'weather_conditions',
    'carriageway_hazards',
    'road_type',
    'road_surface_conditions',
    'speed_limit',         # נוסף
    'light_conditions',    # נוסף
    'urban_or_rural_area'  # נוסף
]

print("📊 בדיקת הערך השכיח (Mode) לכל העמודות המיועדות להשלמה:")
print("-" * 65)

for col in all_low_null_cols:
    if col in df.columns:
        # מוצאים את הערך הנפוץ ביותר שהוא לא אחד מהקודים של Unknown
        actual_data = df[~df[col].isin([9, 99, -1, 6])]

        if not actual_data.empty:
            mode_val = actual_data[col].mode()[0]
            count_mode = (df[col] == mode_val).sum()
            pct_mode = (count_mode / len(df)) * 100

            print(f"🔹 עמודה: {col}")
            print(f"   - הערך השכיח: {mode_val}")
            print(f"   - אחוז באוכלוסייה: {pct_mode:.2f}%")
            print("-" * 35)

print("\n💡 דולב, שים לב במיוחד ל-3 האחרונות:")
print("אם ב-speed_limit השכיח הוא 30, זה אומר שבתאונות ה'לא ידועות' נשלים ל-30.")

# 1. הגדרת מיפוי הערכים לפי קובץ הקידוד של STATS19
road_type_mapping = {
    1: "Roundabout (כיכר)",
    2: "One way street (רחוב חד-סטרי)",
    3: "Dual carriageway (כביש דו-מסלולי)",
    6: "Single carriageway (כביש חד-מסלולי)",
    7: "Slip road (נתיב השתלבות/יציאה)",
    9: "Unknown (לא ידוע)",
    -1: "Data missing (מידע חסר)"
}

# 2. חישוב הכמויות והאחוזים
counts = df['road_type'].value_counts()
percentages = df['road_type'].value_counts(normalize=True) * 100

# 3. יצירת טבלת סיכום מסודרת
distribution_df = pd.DataFrame({
    'Code': counts.index,
    'Description': [road_type_mapping.get(code, "Other") for code in counts.index],
    'Count': counts.values,
    'Percentage (%)': percentages.values
})

print("📊 התפלגות מלאה של סוג הדרך (road_type):")
print("-" * 60)
print(distribution_df.to_string(index=False))

# בונוס: הדפסת הערך השכיח (Mode) כפי שביקשת קודם
actual_mode = df[~df['road_type'].isin([9, -1])]['road_type'].mode()[0]
print("-" * 60)
print(f"💡 הערך השכיח (ללא Unknown) הוא: {actual_mode} ({road_type_mapping.get(actual_mode)})")