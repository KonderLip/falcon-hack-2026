import math
import random

import pandas as pd


def train_val_split(
    df: pd.DataFrame,
    val_size: float,
    seed: int = 56,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)

    vehicle_ids = df["vehicle_id"].unique()
    rng.shuffle(vehicle_ids)

    n_val = math.ceil(val_size * len(vehicle_ids))
    mask = df["vehicle_id"].isin(vehicle_ids[:n_val])

    train_df = df[~mask].reset_index(drop=True)
    val_df = df[mask].reset_index(drop=True)

    return train_df, val_df


def kfold_split(
    df: pd.DataFrame,
    n_splits: int,
    seed: int = 56,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    rng = random.Random(seed)

    vehicle_ids = df["vehicle_id"].unique()
    rng.shuffle(vehicle_ids)

    n_val = math.ceil(len(vehicle_ids) / n_splits)

    folds = []

    for i in range(n_splits):
        val_ids = vehicle_ids[i * n_val:(i + 1) * n_val]
        mask = df["vehicle_id"].isin(val_ids)

        train_df = df[~mask].reset_index(drop=True)
        val_df = df[mask].reset_index(drop=True)

        folds.append((train_df, val_df))

    return folds


def query_gallery_split(
    df: pd.DataFrame,
    seed: int = 56,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)

    queries = []
    gallery = []

    for vehicle_id, sub_df in df.groupby("vehicle_id"):
        cameras = sub_df["camera_id"].unique()
        rng.shuffle(cameras)

        assert len(cameras) >= 2
        n_gallery = 1

        mask = sub_df["camera_id"].isin(cameras[:n_gallery])
        queries.append(sub_df[~mask])
        gallery.append(sub_df[mask])

    query_df = pd.concat(queries, ignore_index=True)
    gallery_df = pd.concat(gallery, ignore_index=True)

    return query_df, gallery_df
