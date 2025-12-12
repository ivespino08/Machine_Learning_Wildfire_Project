import numpy as np
from config import PROCESSED_X_PATH, PROCESSED_Y_PATH, ID_COL
import os
import pandas as pd
from preprocess import load_raw_daily, build_positive_windows, build_negative_windows, build_compressed_dataset
from train import run_training

def run_preprocess():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading raw daily data...")
    df = load_raw_daily()
    print(f"Raw shape: {df.shape}")

    print("Building ignition-centered (positive) 75-day windows...")
    pos_windowed_df = build_positive_windows(df)
    n_pos = pos_windowed_df[ID_COL].nunique()
    print(f"Positive windowed shape (rows): {pos_windowed_df.shape}")
    print(f"Number of positive samples: {n_pos}")

    print("Building negative 75-day windows (no ignition in window)...")
    neg_windowed_df = build_negative_windows(
        df,
        n_negative_target=n_pos,
        start_sample_id=n_pos,
    )
    n_neg = neg_windowed_df[ID_COL].nunique()
    print(f"Negative windowed shape (rows): {neg_windowed_df.shape}")
    print(f"Number of negative samples: {n_neg}")

    windowed_df = pd.concat([pos_windowed_df, neg_windowed_df], ignore_index=True)
    print(f"Total windowed shape (rows): {windowed_df.shape}")
    print(f"Total samples: {windowed_df[ID_COL].nunique()}")

    print("Compressing windows with per-feature k-means...")
    X, y = build_compressed_dataset(windowed_df)
    print(f"Compressed X shape: {X.shape}")
    print(f"Labels y shape: {y.shape}, positive rate={y.mean():.4f}")

    print(f"Saving to {PROCESSED_X_PATH} and {PROCESSED_Y_PATH}...")
    np.save(PROCESSED_X_PATH, X)
    np.save(PROCESSED_Y_PATH, y)

    print("Done.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py [preprocess|train|analyze]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "preprocess":
        run_preprocess()
    elif mode == "train":
        run_training()
    else:
        raise ValueError(f"Unknown command: {mode}")