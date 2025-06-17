import os
import glob
import random
import pandas as pd

def collect_segments(root_dir):
    data = []
    for pain_type, label in [('low_pain', 'PL_1'), ('med_pain', 'PL_2')]:
        pain_dir = os.path.join(root_dir, pain_type)
        if not os.path.isdir(pain_dir):
            continue
        for subject_id in os.listdir(pain_dir):
            subject_dir = os.path.join(pain_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            segment_files = glob.glob(os.path.join(subject_dir, '*.mat'))
            for seg_path in segment_files:
                data.append({
                    'subject_id': subject_id,
                    'segment_path': seg_path,
                    'pain_label': label
                })
    return pd.DataFrame(data)

def split_subjects(df, train_ratio=0.8, seed=42):
    unique_subjects = sorted(df['subject_id'].unique())
    random.seed(seed)
    random.shuffle(unique_subjects)
    n_train = int(len(unique_subjects) * train_ratio)
    train_subjects = unique_subjects[:n_train]
    test_subjects = unique_subjects[n_train:]
    df_train = df[df['subject_id'].isin(train_subjects)].reset_index(drop=True)
    df_test = df[df['subject_id'].isin(test_subjects)].reset_index(drop=True)
    return df_train, df_test, train_subjects, test_subjects

if __name__ == '__main__':
    root_dir = r'D:\bio_s'  # Change this if needed
    df = collect_segments(root_dir)
    print(f'Total segments found: {len(df)}')
    print(f"Subjects: {df['subject_id'].nunique()} unique")
    df_train, df_test, train_subjects, test_subjects = split_subjects(df)
    print(f'Train subjects: {len(train_subjects)}, Train segments: {len(df_train)}')
    print(f'Test subjects: {len(test_subjects)}, Test segments: {len(df_test)}')
    # Optionally save to CSV for inspection:
    # df_train.to_csv('train_segments.csv', index=False)
    # df_test.to_csv('test_segments.csv', index=False)