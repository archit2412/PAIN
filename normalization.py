import pandas as pd
import numpy as np

def zscore_normalize_segment(segment_features):
    """
    Takes a dict of features for one segment. Returns a new dict with z-score normalized features (per segment).
    z = (x - mean) / std for all values in the feature vector.
    """
    values = np.array(list(segment_features.values()), dtype=float)
    mu = values.mean()
    sigma = values.std()
    # Avoid division by zero
    if sigma == 0:
        norm_values = np.zeros_like(values)
    else:
        norm_values = (values - mu) / sigma
    return dict(zip(segment_features.keys(), norm_values))

def normalize_features_batch(features_list):
    """
    Takes a list of feature dicts (one per segment).
    Returns a pandas DataFrame of normalized features (z-score per segment).
    """
    normalized = [zscore_normalize_segment(f) for f in features_list]
    df = pd.DataFrame(normalized)
    return df

# Example usage:
# features_list = [extract_features(seg, fs, labels) for seg in all_segments]
# norm_df = normalize_features_batch(features_list)