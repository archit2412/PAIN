import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import entropy, skew, kurtosis
import pywt

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def log_rms(signal):
    return np.log1p(rms(signal))

def std(signal):
    return np.std(signal)

def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal)))

def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

def slope(signal):
    return (signal[-1] - signal[0]) / len(signal)

def auc(signal):
    return np.trapz(np.abs(signal))

def log_feature(signal):
    return np.log1p(np.abs(signal)).mean()

def scl_features(signal, fs):
    # Mean SCL
    mean_scl = np.mean(signal)
    # Slope
    scl_slope = slope(signal)
    # AUC
    scl_auc = auc(signal)
    # Log mean
    scl_log = log_feature(signal)
    # SCR detection (simple thresholding approach)
    scrs, _ = find_peaks(signal, height=mean_scl + np.std(signal))
    scr_amplitude = signal[scrs].mean() if len(scrs) > 0 else 0
    return {
        "scl_mean": mean_scl,
        "scl_slope": scl_slope,
        "scl_auc": scl_auc,
        "scl_log": scl_log,
        "scr_count": len(scrs),
        "scr_amplitude": scr_amplitude,
    }

def ecg_hr(signal, fs):
    # Simple R-peak detection using find_peaks
    peaks, _ = find_peaks(signal, distance=fs*0.6, height=np.percentile(signal, 95))
    if len(peaks) < 2:
        return {
            "hr": 0, "rr_interval": 0, "rmssd": 0, "sdnn": 0,
            "lf": 0, "hf": 0, "lf_hf": 0, "sd1": 0, "sd2": 0, "log_lf_hf": 0
        }
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr) if np.mean(rr) > 0 else 0
    rr_interval = np.mean(rr)
    # HRV time domain
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) if len(rr) > 1 else 0
    sdnn = np.std(rr)
    # Poincaré
    if len(rr) > 1:
        rr_diff = np.diff(rr)
        sd1 = np.sqrt(np.std(rr_diff) ** 2 / 2)
        sd2 = np.sqrt(2 * np.std(rr) ** 2 - 0.5 * np.std(rr_diff) ** 2)
    else:
        sd1 = 0
        sd2 = 0
    # Frequency domain
    if len(rr) > 2:
        fxx, pxx = welch(rr, fs=1.0/np.mean(rr), nperseg=min(256, len(rr)))
        lf_band = (fxx >= 0.04) & (fxx < 0.15)
        hf_band = (fxx >= 0.15) & (fxx < 0.4)
        lf = np.trapz(pxx[lf_band], fxx[lf_band]) if np.any(lf_band) else 0
        hf = np.trapz(pxx[hf_band], fxx[hf_band]) if np.any(hf_band) else 0
        lf_hf = lf / hf if hf > 0 else 0
        log_lf_hf = np.log1p(lf_hf)
    else:
        lf = hf = lf_hf = log_lf_hf = 0
    return {
        "hr": hr, "rr_interval": rr_interval, "rmssd": rmssd, "sdnn": sdnn,
        "lf": lf, "hf": hf, "lf_hf": lf_hf, "sd1": sd1, "sd2": sd2, "log_lf_hf": log_lf_hf
    }

def wavelet_features(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Energy per sub-band
    energies = [np.sum(np.square(c)) for c in coeffs]
    entropy_feats = [entropy(np.abs(c) / (np.sum(np.abs(c)) + 1e-8)) for c in coeffs]
    skewness = [skew(c) for c in coeffs]
    kurt = [kurtosis(c) for c in coeffs]
    res = {}
    for i, (e, ent, sk, ku) in enumerate(zip(energies, entropy_feats, skewness, kurt)):
        res[f"w_energy_l{i}"] = e
        res[f"w_entropy_l{i}"] = ent
        res[f"w_skew_l{i}"] = sk
        res[f"w_kurt_l{i}"] = ku
    return res

def extract_features(segment_data, fs, labels):
    """
    segment_data: np.ndarray shape [n_samples, n_channels]
    fs: int, sample rate
    labels: list of str, length n_channels
    Returns: dict of all features for this segment
    """
    feats = {}
    # EMG features
    for ch, label in enumerate(labels):
        sig = segment_data[:, ch]
        if 'corrugator' in label.lower() or 'zygomaticus' in label.lower() or 'trapezius' in label.lower() or label.lower().startswith('emg'):
            feats[f"{label}_rms"] = rms(sig)
            feats[f"{label}_log_rms"] = log_rms(sig)
            feats[f"{label}_std"] = std(sig)
            feats[f"{label}_waveform_length"] = waveform_length(sig)
            feats[f"{label}_zcr"] = zero_crossing_rate(sig)
            feats.update({f"{label}_{k}": v for k, v in wavelet_features(sig).items()})
        elif 'scl' in label.lower() or 'gsc' in label.lower() or 'eda' in label.lower():
            scl = scl_features(sig, fs)
            feats.update({f"{label}_{k}": v for k, v in scl.items()})
            feats.update({f"{label}_{k}": v for k, v in wavelet_features(sig).items()})
        elif 'ecg' in label.lower():
            ecg = ecg_hr(sig, fs)
            feats.update({f"{label}_{k}": v for k, v in ecg.items()})
            feats.update({f"{label}_{k}": v for k, v in wavelet_features(sig).items()})
        else:
            # For any other channel, just extract wavelet features
            feats.update({f"{label}_{k}": v for k, v in wavelet_features(sig).items()})
    return feats