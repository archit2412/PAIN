import numpy as np
import pywt

def wavelet_denoise(signal, wavelet='db4', level=None, thresholding='soft', threshold_scale=1.0):
    """
    Perform wavelet denoising on a 1D signal.
    
    Parameters:
        signal (np.ndarray): 1D input signal to denoise.
        wavelet (str): Wavelet type to use (e.g., 'db4', 'sym5').
        level (int or None): Decomposition level. If None, calculated automatically.
        thresholding (str): 'soft' or 'hard' thresholding.
        threshold_scale (float): Multiplier for universal threshold (default: 1.0).
    
    Returns:
        np.ndarray: Denoised 1D signal (same shape as input).
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [coeffs[0]]  # Keep approximation untouched
    for c in coeffs[1:]:
        new_c = pywt.threshold(c, value=uthresh, mode=thresholding)
        new_coeffs.append(new_c)
    denoised = pywt.waverec(new_coeffs, wavelet=wavelet)
    return denoised[:len(signal)]  # In case of padding

def denoise_multichannel(data, wavelet='db4', level=None, thresholding='soft', threshold_scale=1.0):
    """
    Apply wavelet denoising to each channel of a multi-channel segment.
    
    Parameters:
        data (np.ndarray): Shape (n_samples, n_channels).
        wavelet (str): Wavelet type.
        level (int or None): Decomposition level.
        thresholding (str): 'soft' or 'hard'.
        threshold_scale (float): Threshold scale multiplier.
    
    Returns:
        np.ndarray: Denoised signals, same shape as input.
    """
    n_channels = data.shape[1]
    denoised_data = np.zeros_like(data)
    for ch in range(n_channels):
        denoised_data[:, ch] = wavelet_denoise(
            data[:, ch], 
            wavelet=wavelet, 
            level=level, 
            thresholding=thresholding,
            threshold_scale=threshold_scale
        )
    return denoised_data

# Example usage:
# denoised = denoise_multichannel(segment_data, wavelet='db4', level=3)