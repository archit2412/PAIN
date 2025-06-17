import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample

def load_mat_segment(path):
    """
    Loads a biosignal segment from a .mat file.
    Returns: data (np.ndarray shape [n_samples, n_channels]), fs (int), labels (list of str)
    """
    mat = loadmat(path)
    data = mat['data']  # shape (n_samples, n_channels)
    fs = int(mat['fs'].squeeze())
    labels = [str(l[0]) if isinstance(l, np.ndarray) else str(l) for l in mat['labels'].squeeze()]
    return data, fs, labels

def bandpass_filter(signal, fs, low, high, order=4):
    """
    Bandpass filter for 1D or 2D signal.
    """
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return np.array([filtfilt(b, a, signal[:, ch]) for ch in range(signal.shape[1])]).T

def notch_filter(signal, fs, freq=50, Q=30):
    """
    Notch filter for removing powerline noise.
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q)
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return np.array([filtfilt(b, a, signal[:, ch]) for ch in range(signal.shape[1])]).T

def downsample_signal(signal, orig_fs, target_fs=250):
    """
    Downsample signal from orig_fs to target_fs.
    """
    if orig_fs == target_fs:
        return signal
    n_samples = int(signal.shape[0] * target_fs / orig_fs)
    if signal.ndim == 1:
        return resample(signal, n_samples)
    else:
        return resample(signal, n_samples, axis=0)