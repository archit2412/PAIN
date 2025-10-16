"""
Step 3: Stratified Train/Test Split for low_pain and medium_pain
"""
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = r'D:\X-ITE Pain'  # Change if necessary
FEATURES_PATH = f"{BASE_DIR}\\advanced_features.csv"
TRAIN_PATH = f"{BASE_DIR}\\train_features.csv"
TEST_PATH = f"{BASE_DIR}\\test_features.csv"

df = pd.read_csv(FEATURES_PATH)
df = df.dropna()
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['pain_level'],
    random_state=42,
)
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print(f"Train set: {len(train_df)} samples | Test set: {len(test_df)} samples")
print(f"Saved: {TRAIN_PATH} and {TEST_PATH}")
