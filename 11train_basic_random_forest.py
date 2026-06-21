import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# 1. הגדרת נתיב וטעינת הקובץ המדויק
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model.csv')
df = pd.read_csv(file_path)

# 2. הפרדה למאפיינים ועמודת מטרה (Target)
target_col = 'collision_severity'
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. פיצול הנתונים לפני כל מניפולציה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 4. הגדרת מודל Random Forest בסיסי (Baseline)
# n_jobs=-1 משתמש בכל ליבות המעבד להאצת האימון
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)

# 5. חיזוי על סט הבדיקה
y_pred = rf_baseline.predict(X_test)
y_prob = rf_baseline.predict_proba(X_test)

# 6. הצגת מדדי בדיקה מקיפים
print("--- Baseline Model Metrics ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 7. ויזואליזציה של מטריצת בלבול
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Baseline')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 8. גרף ROC-AUC רב-מחלקתי
plt.figure(figsize=(10, 8))
classes = rf_baseline.classes_
for i in range(len(classes)):
    # חישוב עבור כל מחלקה בנפרד (One-vs-Rest)
    fpr, tpr, _ = roc_curve(y_test == classes[i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Severity {classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()