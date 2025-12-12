import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple
from joblib import Parallel, delayed
from config import (
    WINDOW_LENGTH,
    TIME_COL,
    ID_COL,
    TARGET_COL,
    RANDOM_STATE,
    KAGGLE_DATASET_ID,
    KAGGLE_FILE_NAME,
    TIME_COL,
    TIME_SERIES_FEATURE_COLS,
    STATIC_COLS,
    KMEANS_K,
    TARGET_COL,
    WINDOW_LENGTH
)

def load_raw_daily() -> pd.DataFrame:
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET_ID,
        KAGGLE_FILE_NAME,
    )

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    return df

def build_positive_windows(df: pd.DataFrame) -> pd.DataFrame:
    assert WINDOW_LENGTH == 75, "This function assumes a 75-day window."

    df_sorted = df.sort_values(["latitude", "longitude", TIME_COL])

    all_windows = []
    sample_id_counter = 0

    for (lat, lon), g in df_sorted.groupby(["latitude", "longitude"], sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        n = len(g)

        wildfire_bool = g[TARGET_COL].astype(str).str.strip().str.lower().eq("yes")

        for center_idx in range(60, n - 14):
            if not wildfire_bool.iloc[center_idx]:
                continue

            start = center_idx - 60
            end = center_idx + 14  # inclusive

            window = g.iloc[start:end + 1].copy()
            if len(window) != WINDOW_LENGTH:
                continue

            window[ID_COL] = sample_id_counter
            all_windows.append(window)
            sample_id_counter += 1

    if not all_windows:
        raise ValueError("No ignition-based windows created. Check labels and coverage.")

    windowed_df = pd.concat(all_windows, ignore_index=True)
    return windowed_df


def build_negative_windows(df: pd.DataFrame, n_negative_target: int, start_sample_id: int = 0, random_state: int = RANDOM_STATE,) -> pd.DataFrame:

    assert WINDOW_LENGTH == 75, "This function assumes a 75-day window."

    rng = np.random.default_rng(random_state)

    df_sorted = df.sort_values(["latitude", "longitude", TIME_COL])

    all_windows = []
    sample_id_counter = start_sample_id

    for (lat, lon), g in df_sorted.groupby(["latitude", "longitude"], sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        n = len(g)

        wildfire_bool = g[TARGET_COL].astype(str).str.strip().str.lower().eq("yes")

        candidate_indices = np.arange(60, n - 14)

        rng.shuffle(candidate_indices)

        for center_idx in candidate_indices:
            if wildfire_bool.iloc[center_idx]:
                continue

            start = center_idx - 60
            end = center_idx + 14

            if wildfire_bool.iloc[start:end + 1].any():
                continue

            window = g.iloc[start:end + 1].copy()
            if len(window) != WINDOW_LENGTH:
                continue

            window[ID_COL] = sample_id_counter
            all_windows.append(window)
            sample_id_counter += 1

            if sample_id_counter - start_sample_id >= n_negative_target:
                break

        if sample_id_counter - start_sample_id >= n_negative_target:
            break

    if not all_windows:
        raise ValueError("No negative windows created. Check data and logic.")

    neg_windowed_df = pd.concat(all_windows, ignore_index=True)
    return neg_windowed_df


def build_sample_groups(windowed_df: pd.DataFrame) -> dict:
    groups = {}
    for sample_id, g in windowed_df.groupby(ID_COL):
        g = g.sort_values(TIME_COL)
        if len(g) != WINDOW_LENGTH:
            continue
        groups[sample_id] = g
    return groups

def compress_series_kmeans(values: np.ndarray, k: int = KMEANS_K, random_state: int = RANDOM_STATE) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1, 1)

    if np.isnan(x).all():
        return np.zeros(k, dtype=float)

    unique_vals = np.unique(x[~np.isnan(x)])
    num_unique = len(unique_vals)

    k_eff = min(k, num_unique, len(x))
    if k_eff <= 0:
        return np.zeros(k, dtype=float)

    kmeans = KMeans(
        n_clusters=k_eff,
        n_init=1,
        random_state=random_state,
    )
    kmeans.fit(x)

    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)
    centers_sorted = centers[order]

    if k_eff < k:
        pad = np.full(k - k_eff, centers_sorted[-1])
        centers_sorted = np.concatenate([centers_sorted, pad])

    return centers_sorted


def compress_window_kmeans(window_df) -> np.ndarray:
    features = []

    for col in TIME_SERIES_FEATURE_COLS:
        series_values = window_df[col].values
        compressed = compress_series_kmeans(series_values)
        features.extend(compressed.tolist())

    static_vals = window_df[STATIC_COLS].iloc[0].values.astype(float)
    features.extend(static_vals.tolist())

    return np.array(features, dtype=float)

def encode_wildfire_label(value) -> int:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("yes", "y", "1", "true"):
            return 1
        if v in ("no", "n", "0", "false"):
            return 0
    try:
        return int(value)
    except Exception:
        raise ValueError(f"Unexpected Wildfire label: {value!r}")


def _compress_one_sample(sample_id, window_df):
    if len(window_df) != WINDOW_LENGTH:
        return None

    x_vec = compress_window_kmeans(window_df)

    center_idx = 60
    raw_label = window_df[TARGET_COL].iloc[center_idx]
    y_val = encode_wildfire_label(raw_label)

    return x_vec, y_val


def build_compressed_dataset(windowed_df) -> Tuple[np.ndarray, np.ndarray]:
    sample_groups = build_sample_groups(windowed_df)
    items = list(sample_groups.items())

    results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(_compress_one_sample)(sample_id, window_df)
        for sample_id, window_df in items
    )

    results = [r for r in results if r is not None]

    X_list, y_list = zip(*results)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)

    return X, y
