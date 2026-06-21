import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

# 1. טעינה
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model_With_Extra_Fatals.csv')
df = pd.read_csv(file_path)

# 2. הכנה (Target: collision_severity)
target_col = 'collision_severity'
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. פיצול (סט הבדיקה נשאר מקורי)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 4. ביצוע Downsampling ל-150,000 (כפי שבוצע בהרצה הקודמת)
train_data = pd.concat([X_train, y_train], axis=1)
fatal_train = train_data[train_data[target_col] == 1]
serious_train = train_data[train_data[target_col] == 2]
slight_train = train_data[train_data[target_col] == 3]

slight_sampled = resample(slight_train, replace=False, n_samples=150000, random_state=42)

train_final = pd.concat([fatal_train, serious_train, slight_sampled])
X_train_final = train_final.drop(columns=[target_col])
y_train_final = train_final[target_col]

# --- 5. הגדרת ענישה אנליטית ביחס הפוך למחלקות ---
# מחשבים את המשקולות לפי היחס המדויק של מחלקת הרוב (3) מול השאר
custom_weights = {
    1: 28.6,  # פיצוי על מחסור של פי 28.6 בקטלניות
    2: 1.95,  # פיצוי על מחסור של פי 1.95 בקשות
    3: 1.0    # בסיס מחלקת הרוב
}

# 6. אימון המודל
print(f"מאמן מודל עם ענישה אנליטית מחושבת: {custom_weights}")
rf_analytical = RandomForestClassifier(n_estimators=100,
                                       class_weight=custom_weights,
                                       random_state=42,
                                       n_jobs=-1)
rf_analytical.fit(X_train_final, y_train_final)

# 7. חיזוי והערכה
y_pred = rf_analytical.predict(X_test)

print("\n--- Analytical Weighted Model Results ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# מטריצת בלבול בפורמט טבלה
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Fatal(1)', 'Actual Serious(2)', 'Actual Slight(3)'],
                         columns=['Pred Fatal(1)', 'Pred Serious(2)', 'Pred Slight(3)'])
print("\nמטריצת בלבול:")
print(cm_df)

