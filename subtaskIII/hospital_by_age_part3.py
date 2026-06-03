import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- קריאת הנתונים ---
df = pd.read_csv("cleaned.csv")
labels = pd.read_csv("train.labels.1.csv")

# הוספת עמודת גודל הגידול
df['tumor_size'] = labels['אבחנה-Tumor size']

# סינון נתונים חסרים
df = df.dropna(subset=['tumor_size', 'age', 'hospital_code'])
# הסרת 5% הגבוהים ביותר בגודל הגידול
df = df[df['tumor_size'] <= df['tumor_size'].quantile(0.95)]

# Categorize age into English-labeled groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 45, 65, np.inf],
    labels=['Young', 'Middle-aged', 'Older']
)

# Optionally: remove top 5% outliers based on tumor size
df = df[df['tumor_size'] <= df['tumor_size'].quantile(0.95)]

# Define age groups
age_groups = ['Young', 'Middle-aged', 'Older']

# Create a single row of 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Plot each age group separately
for ax, group in zip(axes, age_groups):
    subset = df[df['age_group'] == group]
    sns.boxplot(data=subset, x='hospital_code', y='tumor_size', ax=ax)
    ax.set_title(f'Tumor Size by Hospital\nAge Group: {group}')
    ax.set_xlabel('Hospital Code')
    ax.set_ylabel('Tumor Size (mm)')
    ax.tick_params(axis='x', rotation=90)

# Adjust layout
plt.tight_layout()
plt.show()
