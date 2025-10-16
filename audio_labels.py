"""
Step 1: Create labels.csv for low_pain and medium_pain classes
"""
import os
import pandas as pd

BASE_DIR = r'D:\X-ITE Pain'  # Change this if your data is elsewhere
PAIN_CLASSES = {"low_pain", "medium_pain"}  # Only use these classes

rows = []
for pain_level in os.listdir(BASE_DIR):
    if pain_level not in PAIN_CLASSES:
        continue
    pain_path = os.path.join(BASE_DIR, pain_level, 'audio')
    if not os.path.isdir(pain_path):
        continue
    for subject in os.listdir(pain_path):
        subject_path = os.path.join(pain_path, subject)
        if not os.path.isdir(subject_path):
            continue
        for f in os.listdir(subject_path):
            if f.lower().endswith('.wav'):
                rows.append({
                    'pain_level': pain_level,
                    'subject': subject,
                    'file_name': f,
                    'audio_path': os.path.join(subject_path, f)
                })

df = pd.DataFrame(rows)
labels_path = os.path.join(BASE_DIR, 'labels.csv')
df.to_csv(labels_path, index=False)
print(f"Created {labels_path} with {len(df)} rows and columns: {df.columns.tolist()}")
