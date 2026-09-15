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
    train_df = df[~mask].copy()
    val_df = df[mask].copy()

    return train_df, val_df


def query_gallery_split(
    df: pd.DataFrame,
    seed: int = 56,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)

    queries = []
    gallery = []

    for vehicle_id, vehicle in df.groupby("vehicle_id"):
        cameras = vehicle["camera_id"].unique()
        rng.shuffle(cameras)

        n_cameras = len(cameras)
        assert n_cameras >= 2
        n_gallery = 1

        mask = vehicle["camera_id"].isin(cameras[:n_gallery])
        queries.append(vehicle[~mask])
        gallery.append(vehicle[mask])

    query_df = pd.concat(queries, ignore_index=True)
    gallery_df = pd.concat(gallery, ignore_index=True)

    return query_df, gallery_df
