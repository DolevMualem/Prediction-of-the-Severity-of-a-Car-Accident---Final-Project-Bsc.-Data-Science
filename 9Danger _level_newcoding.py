import pandas as pd
import numpy as np
import os

# 1. הגדרת נתיב אוטומטי לשולחן העבודה
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_name = 'Final_Dataset_Cleaned_Ready.csv'  # וודא שהסיומת היא .csv
full_path = os.path.join(desktop_path, file_name)

print(f"מנסה לטעון את הקובץ מ: {full_path}")

try:
    # טעינת הנתונים
    df = pd.read_csv(full_path)
    print("הקובץ נטען בהצלחה! מתחיל בקידוד...")

    # --- א. יצירת משתנים בינאריים (Binary Transformations) ---
    df['is_rural'] = df['urban_or_rural_area'].map({1: 0, 2: 1, 3: 0})
    df['has_road_hazard'] = np.where(df['carriageway_hazards'] == 0, 0, 1)
    df['did_skid'] = df['skidding_and_overturning'].map({0: 0, 1: 1})
    df['hit_previous_accident'] = df['hit_object_in_carriageway'].map({0: 0, 1: 1})

    # --- ב. קידוד אורדינלי (Ordinal Mapping) ---
    df['road_type'] = df['road_type'].map({1: 0, 2: 1, 7: 2, 3: 3, 6: 4})
    df['junction_detail'] = df['junction_detail'].map({0: 0, 18: 1, 19: 2, 13: 3, 16: 4, 17: 4})
    df['pedestrian_crossing'] = df['pedestrian_crossing'].map({0: 0, 17: 1, 13: 2, 14: 3, 15: 3, 11: 4, 12: 4, 16: 4})
    df['light_conditions'] = df['light_conditions'].map({1: 0, 4: 1, 5: 2, 6: 2, 7: 2})
    df['weather_conditions'] = df['weather_conditions'].map({1: 0, 4: 0, 8: 0, 2: 1, 3: 1, 5: 2, 6: 2, 7: 3})
    df['road_surface_conditions'] = df['road_surface_conditions'].map({1: 0, 2: 1, 7: 1, 3: 2, 4: 2, 5: 3, 6: 3})
    df['vehicle_leaving_carriageway'] = df['vehicle_leaving_carriageway'].map({0: 0, 1: 1, 3: 1, 2: 2, 4: 2, 8: 2, 5: 3, 6: 3, 7: 3})
    df['hit_object_off_carriageway'] = df['hit_object_off_carriageway'].map({0: 0, 1: 1, 2: 1, 3: 1, 5: 1, 10: 1, 7: 2, 9: 2, 11: 2, 4: 3, 6: 3, 8: 3})

    # --- ג. ניקוי וסיום ---
    df.dropna(inplace=True)

    # המרת בינאריים ל-Int
    new_features = ['is_rural', 'has_road_hazard', 'did_skid', 'hit_previous_accident']
    df[new_features] = df[new_features].astype(int)

    print("הקידוד הסתיים בהצלחה!")

    # --- בדיקת גודל המודל ---
    print(f"מספר שורות בסט הנתונים הסופי: {df.shape[0]}")
    print(f"מספר עמודות בסט הנתונים הסופי: {df.shape[1]}")

    # שמירת הקובץ החדש בשולחן העבודה
    output_path = os.path.join(desktop_path, 'Dataset_Ready_For_Model.csv')
    df.to_csv(output_path, index=False)
    print(f"הקובץ המעובד נשמר בשולחן העבודה בשם: Dataset_Ready_For_Model.csv")

except FileNotFoundError:
    print(f"שגיאה: הקובץ '{file_name}' לא נמצא בשולחן העבודה. וודא שהשם מדויק.")
except Exception as e:
    print(f"קרתה שגיאה במהלך ההרצה: {e}")

