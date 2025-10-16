"""
Retrain XGBoost on all data and save final model
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

BASE_DIR = r'D:\X-ITE Pain'
TRAIN_PATH = f"{BASE_DIR}\\train_features.csv"
TEST_PATH = f"{BASE_DIR}\\test_features.csv"
FINAL_MODEL_PATH = "xgb_final_trained_all_features.joblib"
FINAL_ENCODER_PATH = "label_encoder_final.joblib"
FINAL_SCALER_PATH = "scaler_final.joblib"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
full_df = pd.concat([train_df, test_df], ignore_index=True)
feature_cols = [col for col in full_df.columns if col not in ['pain_level', 'subject', 'file_name', 'audio_path']]
X_full = full_df[feature_cols].values
y_full = full_df['pain_level']
le = LabelEncoder()
y_full_encoded = le.fit_transform(y_full)
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
final_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
final_model.fit(X_full_scaled, y_full_encoded)
joblib.dump(final_model, FINAL_MODEL_PATH)
joblib.dump(le, FINAL_ENCODER_PATH)
joblib.dump(scaler, FINAL_SCALER_PATH)
print(f"Final model saved as: {FINAL_MODEL_PATH}")
print(f"Label encoder saved as: {FINAL_ENCODER_PATH}")
print(f"Scaler saved as: {FINAL_SCALER_PATH}")
