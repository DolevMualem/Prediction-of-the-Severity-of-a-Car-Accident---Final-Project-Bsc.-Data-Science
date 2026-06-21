import pandas as pd
import os

# 1. טעינת הקובץ משולחן העבודה
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_path = os.path.join(desktop_path, 'Final_Dataset_Cleaned_Ready.csv')

if not os.path.exists(file_path):
    print(f"שגיאה: הקובץ לא נמצא בנתיב {file_path}")
else:
    df = pd.read_csv(file_path, low_memory=False)
    print("✅ הקובץ נטען בהצלחה!")


    # 2. פונקציית EDA למיפוי קטגוריות והתפלגות
    def detailed_column_mapping(df):
        print("\n" + "=" * 60)
        print("📊 דוח מיפוי עמודות והתפלגות קטגוריות")
        print("=" * 60)

        results = []
        for col in df.columns:
            # סטטיסטיקות בסיסיות
            unique_count = df[col].nunique()
            dtype = df[col].dtype
            top_value = df[col].mode()[0]

            # בדיקת התפלגות
            value_counts = df[col].value_counts(normalize=True).head(5) * 100
            dist_str = ", ".join([f"{val}: {pct:.1f}%" for val, pct in value_counts.items()])

            print(f"\n🔹 עמודה: {col}")
            print(f"   - סוג: {dtype} | ערכים ייחודיים: {unique_count}")
            print(f"   - שכיח: {top_value}")
            print(f"   - התפלגות (Top 5): {dist_str}")

            # המלצה ראשונית לטיפול (לוגיקה אוטומטית)
            if unique_count == 2:
                status = "Binary (OK)"
            elif 'age' in col or 'priority' in col or 'severity' in col:
                status = "Ordinal (Keep Order)"
            else:
                status = "Categorical (Need One-Hot?)"

            print(f"   💡 סיווג משוער: {status}")
            print("-" * 40)


    detailed_column_mapping(df)


# --- 3. רעיון לניתוח EDA נוסף: מבחן משמעות לחיזוי (Feature-Target Impact) ---
def analyze_predictive_power(df, target='target_severity'):
    if target in df.columns:
        print("\n" + "=" * 60)
        print(f"🎯 בדיקת 'כוח החיזוי' של העמודות מול {target}")
        print("=" * 60)

        # נבחר כמה עמודות מעניינות לבדיקה
        sample_features = ['speed_limit', 'road_type', 'age_of_vehicle', 'has_young_driver']

        for col in sample_features:
            if col in df.columns:
                # חישוב אחוז התאונות הקשות (1-2) בתוך כל קטגוריה
                impact = df.groupby(col)[target].apply(lambda x: (x.isin([1, 2]).mean()) * 100)
                print(f"\n📈 השפעת {col} על חומרה גבוהה (%):")
                print(impact.round(2))

# analyze_predictive_power(df)