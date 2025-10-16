"""
Step 2: Extract advanced audio features for low_pain and medium_pain samples
"""
import pandas as pd
import numpy as np
import librosa

BASE_DIR = r'D:\X-ITE Pain'  # Change if necessary
labels_path = f"{BASE_DIR}\\labels.csv"
FEATURES_PATH = f"{BASE_DIR}\\advanced_features.csv"
SR = 44100
N_MFCC = 13
N_MELS = 128

def extract_features(audio_path):
    try:
        y, _ = librosa.load(audio_path, sr=SR)
        features = []
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.min(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1))
        # Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))
        # Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=SR)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        # Mel Spectrogram (log-mel)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel)
        features.extend(np.mean(log_mel, axis=1))
        features.extend(np.std(log_mel, axis=1))
        features.extend(np.min(log_mel, axis=1))
        features.extend(np.max(log_mel, axis=1))
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=SR)
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SR)
        features.extend(np.mean(tonnetz, axis=1))
        features.extend(np.std(tonnetz, axis=1))
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
        features.append(np.mean(centroid))
        features.append(np.std(centroid))
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        # RMS energy
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

df = pd.read_csv(labels_path)
features = [extract_features(row['audio_path']) for _, row in df.iterrows()]
feat_df = pd.DataFrame(features, columns=feature_names)
result_df = pd.concat([df, feat_df], axis=1)
result_df.to_csv(FEATURES_PATH, index=False)
print(f"Advanced feature extraction complete. Saved to {FEATURES_PATH} with shape {result_df.shape}")
