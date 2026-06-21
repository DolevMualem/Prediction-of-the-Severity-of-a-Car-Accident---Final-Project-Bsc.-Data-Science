import pandas as pd
import os

# הגדרת נתיבים
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_main = os.path.join(desktop_path, 'filtered_collisions.csv')
file_raw_assets = os.path.join(desktop_path, 'Aggregation_Vehicle_Casualty2.csv')
file_final = os.path.join(desktop_path, 'Final_Dataset_For_Model3.csv')

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