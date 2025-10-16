"""
Late Fusion: Physiological (biofinal) + Audio (Audio_3) with Confusion Matrices and LOSO
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import defaultdict

BASE_DIR = r"D:\X-ITE Pain"
AUDIO_CSV = os.path.join(BASE_DIR, "advanced_features.csv")
BIO_CSV = os.path.join(os.getcwd(), "bio_features.csv")

audio_df = pd.read_csv(AUDIO_CSV)
bio_df = pd.read_csv(BIO_CSV)
common_cols = ["pain_level", "subject"]
le = LabelEncoder()
le.fit(pd.concat([audio_df["pain_level"], bio_df["pain_level"]], axis=0))
exclude_audio = {"pain_level", "subject", "file_name", "audio_path"}
audio_features = [c for c in audio_df.columns if c not in exclude_audio]
bio_features = [c for c in bio_df.columns if c.startswith("bio_f_")]
audio_subjects = audio_df["subject"].astype(str).values
bio_subjects = bio_df["subject"].astype(str).values
subjects_intersection = sorted(set(audio_subjects).intersection(set(bio_subjects)))
assert subjects_intersection, "No common subjects across audio and bio datasets. LOSO fusion requires overlap."
audio_mask = np.isin(audio_subjects, subjects_intersection)
bio_mask = np.isin(bio_subjects, subjects_intersection)
audio_df_i = audio_df.loc[audio_mask].reset_index(drop=True)
bio_df_i = bio_df.loc[bio_mask].reset_index(drop=True)
audio_subjects_i = audio_df_i["subject"].astype(str).values
bio_subjects_i = bio_df_i["subject"].astype(str).values
y_audio = le.transform(audio_df_i["pain_level"])
y_bio = le.transform(bio_df_i["pain_level"])
X_audio = audio_df_i[audio_features].values
X_bio = bio_df_i[bio_features].values
audio_clf = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_estimators=300, max_depth=6, learning_rate=0.03, n_jobs=-1, random_state=42)),
])
bio_clf = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])
# Hold-out fusion
Xa_tr, Xa_te, ya_tr, ya_te, sa_tr, sa_te = train_test_split(
    X_audio, y_audio, audio_subjects_i, test_size=0.2, stratify=y_audio, random_state=42
)
train_subjects = set(sa_tr)
test_subjects = set(sa_te)
bio_train_mask = np.isin(bio_subjects_i, list(train_subjects))
bio_test_mask = np.isin(bio_subjects_i, list(test_subjects))
Xb_tr, yb_tr = X_bio[bio_train_mask], y_bio[bio_train_mask]
Xb_te, yb_te = X_bio[bio_test_mask], y_bio[bio_test_mask]
audio_clf.fit(Xa_tr, ya_tr)
bio_clf.fit(Xb_tr, yb_tr)
pa = audio_clf.predict_proba(Xa_te)
pb = bio_clf.predict_proba(Xb_te)
def subject_mean_proba(subjects, y_true, proba):
    d = {}
    y_map = {}
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        d[s] = proba[idx].mean(axis=0)
        if idx.size:
            vals, cnts = np.unique(y_true[idx], return_counts=True)
            y_map[s] = vals[np.argmax(cnts)]
    return d, y_map
pa_map, ya_map = subject_mean_proba(sa_te, ya_te, pa)
pb_map, yb_map = subject_mean_proba(bio_subjects_i[bio_test_mask], yb_te, pb)
fusion_subjects = sorted(set(pa_map.keys()).intersection(pb_map.keys()))
proba_fused = np.vstack([(pa_map[s] + pb_map[s]) / 2.0 for s in fusion_subjects])
y_true_fused = np.array([ya_map.get(s, yb_map[s]) for s in fusion_subjects])
y_pred_fused = np.argmax(proba_fused, axis=1)
print("\n==============================")
print("Hold-out Fusion Confusion Matrix (Fusion Model)")
cm_holdout = confusion_matrix(y_true_fused, y_pred_fused)
print(pd.DataFrame(cm_holdout, index=[f"True_{c}" for c in le.classes_], columns=[f"Pred_{c}" for c in le.classes_]))
print("\nHold-out Fusion Classification Report:")
print(classification_report(y_true_fused, y_pred_fused, target_names=le.classes_))
print("==============================\n")
# LOSO fusion
logo = LeaveOneGroupOut()
cm_total = np.zeros((len(le.classes_), len(le.classes_)), dtype=int)
accs = []
for train_idx, test_idx in logo.split(X_audio, y_audio, groups=audio_subjects_i):
    test_subjects_fold = set(audio_subjects_i[test_idx])
    bio_tr_mask = ~np.isin(bio_subjects_i, list(test_subjects_fold))
    bio_te_mask = np.isin(bio_subjects_i, list(test_subjects_fold))
    Xb_tr, yb_tr = X_bio[bio_tr_mask], y_bio[bio_tr_mask]
    Xb_te, yb_te = X_bio[bio_te_mask], y_bio[bio_te_mask]
    sb_te = bio_subjects_i[bio_te_mask]
    Xa_tr, Xa_te = X_audio[train_idx], X_audio[test_idx]
    ya_tr, ya_te = y_audio[train_idx], y_audio[test_idx]
    sa_te = audio_subjects_i[test_idx]
    if Xa_tr.size == 0 or Xb_tr.size == 0 or Xa_te.size == 0 or Xb_te.size == 0:
        continue
    a_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_estimators=300, max_depth=6, learning_rate=0.03, n_jobs=-1, random_state=42)),
    ])
    b_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    a_model.fit(Xa_tr, ya_tr)
    b_model.fit(Xb_tr, yb_tr)
    pa = a_model.predict_proba(Xa_te)
    pb = b_model.predict_proba(Xb_te)
    pa_map, ya_map = subject_mean_proba(sa_te, ya_te, pa)
    pb_map, yb_map = subject_mean_proba(sb_te, yb_te, pb)
    fuse_subjects = sorted(set(pa_map.keys()).intersection(pb_map.keys()))
    if not fuse_subjects:
        continue
    pf = np.vstack([(pa_map[s] + pb_map[s]) / 2.0 for s in fuse_subjects])
    y_true = np.array([ya_map.get(s, yb_map[s]) for s in fuse_subjects])
    y_pred = np.argmax(pf, axis=1)
    cm_total += confusion_matrix(y_true, y_pred, labels=np.arange(len(le.classes_)))
    accs.append(accuracy_score(y_true, y_pred))
print("LOSO fusion mean accuracy:", np.mean(accs) if accs else 0.0)
print("\n==============================")
print("LOSO Fusion Confusion Matrix (Fusion Model)")
print(pd.DataFrame(cm_total, index=[f"True_{c}" for c in le.classes_], columns=[f"Pred_{c}" for c in le.classes_]))
print("==============================\n")
