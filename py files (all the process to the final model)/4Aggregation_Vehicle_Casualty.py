import pandas as pd
import os

# 1. הגדרת נתיבים
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
input_file = os.path.join(desktop_path, 'merged_vehicle_casualty_clean2.csv')
output_file = os.path.join(desktop_path, 'Aggregation_Vehicle_Casualty2.csv')

# טעינת הנתונים
if not os.path.exists(input_file):
    print(f"שגיאה: הקובץ {input_file} לא נמצא על שולחן העבודה.")
else:
    df = pd.read_csv(input_file)

    # עבודה על עותק כדי לא לגעת ב-df המקורי בזיכרון
    df_working = df.copy()

    # --- שלב א': הכנת עמודות (Pre-processing) לפני אגריגציה ---

    # א. ניקוי עמודות (הפיכת 9/99 ל-1-)
    df_working['vehicle_leaving_carriageway'] = df_working['vehicle_leaving_carriageway'].replace(9, -1)
    df_working['hit_object_off_carriageway'] = df_working['hit_object_off_carriageway'].replace(99, -1)

    # ב. הפיכת משתנים לבינאריים (0/1)
    df_working['escooter_flag'] = df_working['escooter_flag'].fillna(0).astype(int)
    df_working['skidding_and_overturning'] = df_working['skidding_and_overturning'].apply(
        lambda x: 1 if 1 <= x <= 5 else 0)
    df_working['hit_object_in_carriageway'] = df_working['hit_object_in_carriageway'].apply(
        lambda x: 1 if 1 <= x <= 12 else 0)
    df_working['sex_of_driver'] = df_working['sex_of_driver'].apply(lambda x: 1 if x == 2 else 0)

    # ג. פיצול גיל הנהג (age_band_of_driver) ל-2 עמודות בינאריות
    df_working['has_young_driver'] = df_working['age_band_of_driver'].apply(lambda x: 1 if 1 <= x <= 4 else 0)
    df_working['has_elderly_driver'] = df_working['age_band_of_driver'].apply(lambda x: 1 if x == 11 else 0)
    # ג. פיצול גיל הנפגע (age_band_of_casualty) ל-2 עמודות בינאריות
    df_working['has_young_casualty'] = df_working['age_band_of_casualty'].apply(lambda x: 1 if 1 <= x <= 4 else 0)
    df_working['has_elderly_casualty'] = df_working['age_band_of_casualty'].apply(lambda x: 1 if x == 11 else 0)

    # ד. סולמות עדיפות (Priority Scores)
    def get_vehicle_priority(v_type):
        if v_type in [1, 2, 3, 4, 5, 22, 23, 97, 103, 104, 105, 106]: return 3  # דו-גלגלי/פגיע
        if v_type in [10, 11, 18, 20, 21, 98, 113]: return 2  # רכב כבד/אוטובוס
        return 1  # רכב פרטי


    def get_casualty_priority(c_type):
        if c_type == 0: return 4  # הולך רגל
        if c_type in [1, 2, 3, 4, 5, 22, 23, 97, 103, 104, 105, 106]: return 3  # דו-גלגלי
        if c_type in [10, 11, 18, 20, 21, 98, 113]: return 2  # כבד
        return 1  # פרטי


    df_working['vehicle_priority'] = df_working['vehicle_type'].apply(get_vehicle_priority)
    df_working['casualty_priority'] = df_working['casualty_type'].apply(get_casualty_priority)

    # ה. קיבוץ סיכון
    towing_risk = {0: 0, 3: 1, 4: 1, 1: 2, 2: 2, 5: 2}
    df_working['towing_risk_group'] = df_working['towing_and_articulation'].map(towing_risk).fillna(0)

    lane_risk = {0: 0, 6: 1, 1: 2, 2: 2, 4: 3, 5: 3, 9: 3}
    df_working['lane_risk_group'] = df_working['vehicle_location_restricted_lane'].map(lane_risk).fillna(0)

    # --- שלב ב': ביצוע האגריגציה (Group By collision_index) ---

    agg_logic = {
        'collision_year_x': 'first',
        'vehicle_reference': 'count',
        'vehicle_priority': 'max',
        'casualty_priority': 'max',
        'towing_risk_group': 'max',
        'lane_risk_group': 'max',
        'escooter_flag': 'max',
        'has_young_casualty': 'max',
        'has_elderly_casualty': 'max',
        'skidding_and_overturning': 'max',
        'hit_object_in_carriageway': 'max',
        'vehicle_leaving_carriageway': 'max',
        'hit_object_off_carriageway': 'max',
        'sex_of_driver': 'max',
        'age_of_vehicle': 'max',
        'has_young_driver': 'max',
        'has_elderly_driver': 'max',
        'casualty_class': 'max',
        'casualty_severity': 'min'
    }

    # יצירת הטבלה החדשה
    Aggregation_Vehicle_Casualty = df_working.groupby('collision_index').agg(agg_logic).reset_index()

    # שינוי שמות עמודות לבהירות
    Aggregation_Vehicle_Casualty = Aggregation_Vehicle_Casualty.rename(columns={
        'vehicle_reference': 'num_vehicles_involved',
        'casualty_severity': 'target_severity'
    })

    # שמירה לקובץ חדש
    Aggregation_Vehicle_Casualty.to_csv(output_file, index=False)

    # --- שלב ג': הדפסות ניתוח בסיסי ---
    print("-" * 40)
    print("דוח סיכום אגריגציה:")
    print("-" * 40)
    print(f"מספר שורות בטבלת המקור (רכבים/נפגעים): {len(df)}")
    print(f"מספר שורות בטבלה המאוחדת (תאונות): {len(Aggregation_Vehicle_Casualty)}")
    print(f"מספר עמודות בטבלה החדשה: {len(Aggregation_Vehicle_Casualty.columns)}")
    print("\nרשימת העמודות בטבלה החדשה:")
    print(Aggregation_Vehicle_Casualty.columns.tolist())

    print("\nהתפלגות חומרת התאונה (Target):")
    severity_counts = Aggregation_Vehicle_Casualty['target_severity'].value_counts().sort_index()
    severity_mapping = {1: "קטלנית (Fatal)", 2: "קשה (Serious)", 3: "קלה (Slight)"}
    for sev, count in severity_counts.items():
        print(f"  - {severity_mapping.get(sev, sev)}: {count} תאונות")

    print(f"\nממוצע רכבים מעורבים לתאונה: {Aggregation_Vehicle_Casualty['num_vehicles_involved'].mean():.2f}")
    print(f"אחוז תאונות עם נהג צעיר (<25): {Aggregation_Vehicle_Casualty['has_young_driver'].mean() * 100:.1f}%")
    print(f"אחוז תאונות עם נהג מבוגר (75+): {Aggregation_Vehicle_Casualty['has_elderly_driver'].mean() * 100:.1f}%")
    print("-" * 40)
    print(f"אחוז תאונות עם נפגע צעיר (<20): {Aggregation_Vehicle_Casualty['has_young_casualty'].mean() * 100:.1f}%")
    print(f"אחוז תאונות עם נפגע מבוגר (75+): {Aggregation_Vehicle_Casualty['has_elderly_casualty'].mean() * 100:.1f}%")
    print(f"הקובץ נשמר בהצלחה בשם: Aggregation_Vehicle_Casualty2.csv")
