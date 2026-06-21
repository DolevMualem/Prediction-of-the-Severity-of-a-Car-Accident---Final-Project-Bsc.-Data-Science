import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. הגדרת נתיב לקובץ (הקובץ המעובד שיצרת בשלב הקודם)
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
file_path = os.path.join(desktop_path, 'Collision_Dataset_For_Model.csv')

# 2. טעינת הנתונים
df = pd.read_csv(file_path)

# 3. ניתוח מספרי
severity_counts = df['collision_severity'].value_counts().sort_index()
severity_pct = df['collision_severity'].value_counts(normalize=True).sort_index() * 100

# מיפוי שמות לבהירות
severity_labels = {1: 'Fatal (1)', 2: 'Serious (2)', 3: 'Slight (3)'}
labels = [severity_labels[i] for i in severity_counts.index]

print("-" * 30)
print("📊 התפלגות חומרת תאונות (Target):")
for idx, count in severity_counts.items():
    print(f"{severity_labels[idx]}: {count:,} תאונות ({severity_pct[idx]:.2f}%)")
print("-" * 30)

# 4. תצוגה ויזואלית (גרף עמודות)
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.barplot(x=labels, y=severity_counts.values, palette="viridis")

# הוספת כמות מעל כל עמודה
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 9),
                textcoords = 'offset points',
                fontsize=11, fontweight='bold')

plt.title('Distribution of Accident Severity (Target Class)', fontsize=15)
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Number of Accidents', fontsize=12)
plt.show()