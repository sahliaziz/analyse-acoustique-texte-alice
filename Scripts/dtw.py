import numpy as np
import pandas as pd


def df_to_feature_matrix(
    df: pd.DataFrame, phoneme_to_id: dict, timing_weight=1.0
) -> np.ndarray:
    """
    Each phoneme becomes a feature vector: [phoneme_id, onset, duration]

    timing_weight: scales the contribution of timing features relative to
                   phoneme identity. Increase to penalise timing mismatches more.
    """
    if df.empty:
        return np.empty((0, 3), dtype=float)

    phoneme_ids = np.array([phoneme_to_id[p] for p in df["text"]], dtype=float)
    phoneme_ids /= max(len(phoneme_to_id) - 1, 1)

    onsets = np.array(df["tmin"].values, dtype=float)
    durations = np.array((df["tmax"] - df["tmin"]).values, dtype=float)

    max_time = max(onsets.max(), 1e-9)
    max_dur = max(durations.max(), 1e-9)
    onsets /= max_time
    durations /= max_dur

    features = np.stack(
        [phoneme_ids, onsets * timing_weight, durations * timing_weight], axis=1
    )
    return features


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> tuple[float, float, int]:
    """
    Standard DTW with Euclidean local distance.
    Returns the normalised DTW distance (divided by the path length).
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        raise ValueError("DTW requires two non-empty sequences")

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local_cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            D[i, j] = local_cost + min(
                D[i - 1, j],  # insertion
                D[i, j - 1],  # deletion
                D[i - 1, j - 1],
            )  # match / substitution

    i, j = n, m
    path_length = 0
    while i > 0 or j > 0:
        path_length += 1
        candidates = {
            (i - 1, j - 1): D[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            (i - 1, j): D[i - 1, j] if i > 0 else np.inf,
            (i, j - 1): D[i, j - 1] if j > 0 else np.inf,
        }
        i, j = min(candidates, key=lambda k: candidates[k])

    raw_distance = D[n, m]
    normalised_distance = raw_distance / path_length
    return raw_distance, normalised_distance, path_length
