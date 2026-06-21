#07/03/2026
import pandas as pd
import os

# הגדרת נתיב לשולחן העבודה
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# 1. טעינת הקבצים המקוריים
print("טוען את הקבצים המקוריים...")

# נגדיר שעמודות מזהים ושמות מודלים ייקראו תמיד כטקסט כדי למנוע Mixed Types
dtypes_dict = {
    'collision_index': str,
    'generic_make_model': str
}

df_collisions = pd.read_csv(f'{desktop_path}/dft-road-casualty-statistics-collision-last-5-years.csv', dtype={'collision_index': str}, low_memory=False)
df_vehicles = pd.read_csv(f'{desktop_path}/dft-road-casualty-statistics-vehicle-last-5-years.csv', dtype=dtypes_dict, low_memory=False)
df_casualties = pd.read_csv(f'{desktop_path}/dft-road-casualty-statistics-casualty-last-5-years.csv', dtype={'collision_index': str}, low_memory=False)
# 2. הגדרת רשימות העמודות למחיקה (vehicle_reference לא נמחקת!)
drop_cols_collision = [
    'collision_ref_no', 'location_easting_osgr', 'location_northing_osgr',
    'police_force', 'local_authority_ons_district', 'local_authority_highway',
    'local_authority_highway_current', 'first_road_number', 'second_road_class',
    'second_road_number', 'junction_detail_historic', 'carriageway_hazards_historic',
    'did_police_officer_attend_scene_of_accident', 'lsoa_of_accident_location',
    'collision_adjusted_severity_serious', 'collision_adjusted_severity_slight'
]

drop_cols_casualty = [
    'casualty_reference', 'age_of_casualty', 'pedestrian_location', 'pedestrian_movement',
    'car_passenger', 'bus_or_coach_passenger', 'pedestrian_road_maintenance_worker',
    'casualty_imd_decile', 'lsoa_of_casualty', 'casualty_distance_banding',
    'casualty_adjusted_serious', 'casualty_adjusted_slight', 'collision_ref_no'
]

drop_cols_vehicle = [
    'vehicle_manoeuvre_historic', 'vehicle_manoeuvre', 'first_point_of_impact',
    'vehicle_left_hand_drive', 'journey_purpose_of_driver', 'age_of_driver',
    'engine_capacity_cc', 'driver_imd_decile', 'lsoa_of_driver', 'driver_distance_banding',
    'vehicle_location_restricted_lane_historic', 'journey_purpose_of_driver_historic', 'collision_ref_no'

]

# 3. מחיקת העמודות הלא רלוונטיות
print("מסיר עמודות לא רלוונטיות...")
df_collisions_filtered = df_collisions.drop(columns=drop_cols_collision, errors='ignore')
df_casualties_filtered = df_casualties.drop(columns=drop_cols_casualty, errors='ignore')
df_vehicles_filtered = df_vehicles.drop(columns=drop_cols_vehicle, errors='ignore')

# 4. הדפסת כמות העמודות שנותרו (פלט מסודר)
print("\n--- סיכום עמודות לאחר סינון ---")
print(f"טבלת תאונות (Collision): נותרו {df_collisions_filtered.shape[1]} עמודות (מתוך {df_collisions.shape[1]} במקור).")
print(f"טבלת רכבים (Vehicle): נותרו {df_vehicles_filtered.shape[1]} עמודות (מתוך {df_vehicles.shape[1]} במקור).")
print(f"טבלת נפגעים (Casualty): נותרו {df_casualties_filtered.shape[1]} עמודות (מתוך {df_casualties.shape[1]} במקור).")

# 5. שמירת הטבלאות החדשות על שולחן העבודה
print("\nשומר את הקבצים החדשים על שולחן העבודה...")
df_collisions_filtered.to_csv(f'{desktop_path}/filtered_collisions.csv', index=False)
df_vehicles_filtered.to_csv(f'{desktop_path}/filtered_vehicles.csv', index=False)
df_casualties_filtered.to_csv(f'{desktop_path}/filtered_casualties.csv', index=False)

print("השמירה בוצעה בהצלחה! הקבצים החדשים מחכים לך על שולחן העבודה.")