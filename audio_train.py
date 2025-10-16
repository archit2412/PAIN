"""
Step 4: Train and evaluate an XGBoost model for low_pain vs. medium_pain
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE_DIR = r'D:\X-ITE Pain'  # Change if necessary
TRAIN_PATH = f"{BASE_DIR}\\train_features.csv"
TEST_PATH = f"{BASE_DIR}\\test_features.csv"
MODEL_PATH = f"{BASE_DIR}\\xgb_audio_model.joblib"
ENCODER_PATH = f"{BASE_DIR}\\label_encoder.joblib"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
exclude_cols = {'pain_level', 'subject', 'file_name', 'audio_path'}
feature_cols = [c for c in train_df.columns if c not in exclude_cols]
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
le = LabelEncoder()
y_train = le.fit_transform(train_df['pain_level'].values)
y_test = le.transform(test_df['pain_level'].values)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        n_jobs=-1,
        random_state=42
    ))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred_labels))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))
joblib.dump(pipeline, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\nModel saved as {MODEL_PATH}")
print(f"Label encoder saved as {ENCODER_PATH}")
