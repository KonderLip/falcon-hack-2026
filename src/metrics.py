import pandas as pd
import numpy as np


def evaluate(
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    similarity: np.ndarray,
):
    assert similarity.shape == (len(query_df), len(gallery_df))

    q_vehicle = query_df["vehicle_id"].to_numpy()
    q_camera = query_df["camera_id"].to_numpy()

    g_vehicle = gallery_df["vehicle_id"].to_numpy()
    g_camera = gallery_df["camera_id"].to_numpy()

    # Official ReID evaluation:
    # same vehicle AND different camera.
    positive = (
        (q_vehicle[:, None] == g_vehicle[None, :]) & (q_camera[:, None] != g_camera[None, :])
    )

    aps = []

    for q in range(len(query_df)):
        order = np.argsort(-similarity[q])
        relevance = positive[q][order]

        n_positive = relevance.sum()

        # For our ReID validation every query should
        # have at least one positive.
        assert n_positive > 0

        hits = np.cumsum(relevance)
        ranks = np.arange(1, len(relevance) + 1)

        precision = hits / ranks

        ap = (precision * relevance).sum() / n_positive
        aps.append(ap)

    return np.mean(aps)
