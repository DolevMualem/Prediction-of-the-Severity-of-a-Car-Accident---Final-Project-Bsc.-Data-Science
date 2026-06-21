import pandas as pd
import os

desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_main = os.path.join(desktop_path, 'filtered_collisions.csv')
file_raw_assets = os.path.join(desktop_path, 'merged_vehicle_casualty_clean2.csv')

print("טוען נתונים...")
df_main = pd.read_csv(file_main, dtype={'collision_index': str})
df_raw = pd.read_csv(file_raw_assets, dtype={'collision_index': str})

# ניקוי מפתחות
df_main['collision_index'] = df_main['collision_index'].str.strip()
df_raw['collision_index'] = df_raw['collision_index'].str.strip()

# --- שלב א': הכנת כל המשתנים (Preprocessing) בבת אחת ---
print("מכין משתנים...")
df_raw['has_young_driver'] = df_raw['age_band_of_driver'].apply(lambda x: 1 if 1 <= x <= 4 else 0)
df_raw['has_elderly_driver'] = df_raw['age_band_of_driver'].apply(lambda x: 1 if x == 11 else 0)
df_raw['has_young_casualty'] = df_raw['age_band_of_casualty'].apply(lambda x: 1 if 1 <= x <= 4 else 0)
df_raw['has_elderly_casualty'] = df_raw['age_band_of_casualty'].apply(lambda x: 1 if x == 11 else 0)
df_raw['escooter_flag'] = df_raw['escooter_flag'].fillna(0).astype(int)

# פונקציות עזר לסיכונים (מתוך הקוד המקורי שלך)
df_raw['skidding_and_overturning'] = df_raw['skidding_and_overturning'].apply(lambda x: 1 if 1 <= x <= 5 else 0)
df_raw['hit_object_in_carriageway'] = df_raw['hit_object_in_carriageway'].apply(lambda x: 1 if 1 <= x <= 12 else 0)

# --- שלב ב': אגרגציה חכמה על כל העמודות ---
print("מבצע אגרגציה מלאה...")

# נגדיר לוגיקה ספציפית לעמודות קריטיות, ולשאר ניתן 'max' כברירת מחדל
agg_logic = {col: 'max' for col in df_raw.columns if col != 'collision_index'}

# תיקונים ללוגיקה ספציפית
agg_logic['vehicle_reference'] = 'count'  # אנחנו רוצים לדעת כמה רכבים היו
agg_logic['casualty_severity'] = 'min'    # חומרה נקבעת לפי הנפגע הכי קשה (1 הוא הכי קשה)

# ביצוע האגרגציה
df_agg = df_raw.groupby('collision_index').agg(agg_logic).reset_index()

# --- שלב ג': מיזוג (Merge) וניקוי כפילויות ---
print("ממזג טבלאות...")
final_df = pd.merge(df_main, df_agg, on='collision_index', how='inner')

# הסרת עמודות כפולות שנוצרו (כמו collision_year שמופיע בשניהם)
# נזהה עמודות עם סיומת _y ונסיר אותן
cols_to_drop = [c for c in final_df.columns if c.endswith('_y')]
final_df = final_df.drop(columns=cols_to_drop)

# ניקוי סיומת _x מהעמודות שנשארו
final_df.columns = [c[:-2] if c.endswith('_x') else c for c in final_df.columns]

# שמירה
output_path = os.path.join(desktop_path, 'Final_Dataset_For_Model.csv')
final_df.to_csv(output_path, index=False)

print("-" * 40)
print(f"הצלחה! הקובץ הסופי כולל את כל המשתנים.")
print(f"שורות: {len(final_df)}")
print(f"עמודות: {len(final_df.columns)}")
print("-" * 40)
###############_____________________
import pandas as pd
import os

# הגדרת נתיבים
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_main = os.path.join(desktop_path, 'filtered_collisions.csv')
file_raw_assets = os.path.join(desktop_path, 'Aggregation_Vehicle_Casualty2.csv')
file_final = os.path.join(desktop_path, 'Final_Dataset_For_Model.csv')

# טעינת שמות העמודות בלבד
cols_main = pd.read_csv(file_main, nrows=0).columns.tolist()
# טעינת הטבלה המשנית כולל העמודות החדשות שיצרנו בקוד הקודם (Flags)
cols_agg = ['collision_index', 'vehicle_reference', 'escooter_flag', 'has_young_driver',
            'has_elderly_driver', 'has_young_casualty', 'has_elderly_casualty', 'casualty_severity']
# (הערה: אם הוספת עוד עמודות ידנית באגרגציה, הן יופיעו ב-Final)

cols_final = pd.read_csv(file_final, nrows=0).columns.tolist()

print("-" * 60)
print(f"1. עמודות בטבלה הראשית (Collisions): {len(cols_main)}")
print(f"   {cols_main}")

print(f"\n2. עמודות בטבלה הסופית (Final Dataset): {len(cols_final)}")

# זיהוי עמודות שהגיעו מהטבלה המשנית
extra_cols = [c for c in cols_final if c not in cols_main]
print(f"\n3. עמודות שנוספו מהטבלה המשנית (או יצירה ידנית): {len(extra_cols)}")
print(f"   {extra_cols}")

# בדיקה האם יש כפילויות סמויות (שמות דומים)
print("\n4. בדיקת כפילויות פוטנציאליות (החשודות בתוספת ל-49):")
import difflib
for c1 in cols_main:
    matches = difflib.get_close_matches(c1, extra_cols, n=1, cutoff=0.8)
    if matches:
        print(f"   חשד לכפילות: '{c1}' (ראשית) דומה מאוד ל-'{matches[0]}' (משנית)")

print("-" * 60)