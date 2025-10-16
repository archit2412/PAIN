"""
Orchestrator script to run all pain prediction steps using all 20 CPU cores.
"""

import os
import sys
import subprocess
import multiprocessing
import pandas as pd
import numpy as np
"""
CPU-aware orchestrator:
- 12 physical cores, 20 logical threads (per screenshot)
- Use 12 processes for feature extraction (CPU-bound, avoid oversubscription)
- Cap BLAS threads to 1 in workers; keep 20 threads for model training tasks
"""

# CPU configuration per your machine
N_PHYSICAL_PROCESSES = 12   # processes for Pool (matches 12 cores)
N_LOGICAL_THREADS = 20      # threads for threaded ML libs (matches 20 logical)

# Set environment variables for max CPU usage
os.environ["OMP_NUM_THREADS"] = str(N_LOGICAL_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_LOGICAL_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_LOGICAL_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_LOGICAL_THREADS)
os.environ["XGB_NUM_THREADS"] = str(N_LOGICAL_THREADS)

# Helper to run a script and stream output
def run_script(script_name):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run([sys.executable, script_name], cwd=os.getcwd())
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(result.returncode)

# Top-level feature extraction function for multiprocessing
SR = 44100
N_MFCC = 13
N_MELS = 128
def extract_features(audio_path):
    try:
        # Import librosa inside worker to honor per-worker thread caps
        import librosa
        y, _ = librosa.load(audio_path, sr=SR)
        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.min(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1))
        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))
        chroma = librosa.feature.chroma_stft(y=y, sr=SR)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel)
        features.extend(np.mean(log_mel, axis=1))
        features.extend(np.std(log_mel, axis=1))
        features.extend(np.min(log_mel, axis=1))
        features.extend(np.max(log_mel, axis=1))
        contrast = librosa.feature.spectral_contrast(y=y, sr=SR)
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SR)
        features.extend(np.mean(tonnetz, axis=1))
        features.extend(np.std(tonnetz, axis=1))
        centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
        features.append(np.mean(centroid))
        features.append(np.std(centroid))
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        n_features = (N_MFCC*4 + N_MFCC*2 + 12*2 + N_MELS*4 + 7*2 + 6*2 + 2*5)
        return [np.nan] * n_features

feature_names = (
    [f"mfcc_mean_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_std_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_min_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_max_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_delta_mean_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_delta_std_{i+1}" for i in range(N_MFCC)] +
    [f"chroma_mean_{i+1}" for i in range(12)] +
    [f"chroma_std_{i+1}" for i in range(12)] +
    [f"mel_mean_{i+1}" for i in range(N_MELS)] +
    [f"mel_std_{i+1}" for i in range(N_MELS)] +
    [f"mel_min_{i+1}" for i in range(N_MELS)] +
    [f"mel_max_{i+1}" for i in range(N_MELS)] +
    [f"contrast_mean_{i+1}" for i in range(7)] +
    [f"contrast_std_{i+1}" for i in range(7)] +
    [f"tonnetz_mean_{i+1}" for i in range(6)] +
    [f"tonnetz_std_{i+1}" for i in range(6)] +
    ["centroid_mean", "centroid_std"] +
    ["bandwidth_mean", "bandwidth_std"] +
    ["rolloff_mean", "rolloff_std"] +
    ["zcr_mean", "zcr_std"] +
    ["rms_mean", "rms_std"]
)

# Step 1: Create labels.csv (invoked later under __main__)

# Step 2: Extract features (parallelized)
def _worker_init():
    """Limit BLAS threads per worker to avoid oversubscription."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run_audio_features_parallel():
    BASE_DIR = r'D:\X-ITE Pain'
    labels_path = f"{BASE_DIR}\\labels.csv"
    FEATURES_PATH = f"{BASE_DIR}\\advanced_features.csv"
    df = pd.read_csv(labels_path)
    paths = df['audio_path'].tolist()
    # Chunking to reduce task overhead
    chunksize = max(1, len(paths) // (N_PHYSICAL_PROCESSES * 8) or 1)
    with multiprocessing.Pool(N_PHYSICAL_PROCESSES, initializer=_worker_init) as pool:
        features = list(pool.imap(extract_features, paths, chunksize=chunksize))
    feat_df = pd.DataFrame(features, columns=feature_names)
    result_df = pd.concat([df, feat_df], axis=1)
    result_df.to_csv(FEATURES_PATH, index=False)
    print(f"Advanced feature extraction complete. Saved to {FEATURES_PATH} with shape {result_df.shape}")

if __name__ == "__main__":
    # Required on Windows for multiprocessing
    multiprocessing.freeze_support()

    # Step 1: Create labels.csv
    run_script("audio_labels.py")

    # Step 2: Extract features (parallelized)
    run_audio_features_parallel()

    # Step 3: Stratified split
    run_script("audio_split.py")

    # Step 4: Train/evaluate audio model
    run_script("audio_train.py")

    # Step 5: Retrain on all data
    run_script("audio_retrain.py")

    # Step 6: Late fusion (biofinal + audio)
    run_script("fusion_late.py")

    print("\nAll steps completed successfully.")
