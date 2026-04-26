from typing import Iterator

import numpy as np
import pandas as pd


def date_based_ts_split(
    dates: pd.Series,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Date-aware time-series CV. Yields (train_idx, test_idx) for `n_splits`
    folds. Each fold: train = rows with date in earlier slice, test = rows
    in next contiguous date slice. No date appears in both train and test."""
    arr = np.asarray(dates.values if hasattr(dates, "values") else dates)
    unique_dates = np.sort(np.unique(arr))
    n_dates = len(unique_dates)
    fold_size = n_dates // (n_splits + 1)
    if fold_size == 0:
        raise ValueError(f"too few dates ({n_dates}) for {n_splits} splits")
    series = pd.Series(arr)
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_end = min(train_end + fold_size, n_dates)
        train_dates = set(unique_dates[:train_end])
        test_dates = set(unique_dates[train_end:test_end])
        train_mask = series.isin(train_dates).to_numpy()
        test_mask = series.isin(test_dates).to_numpy()
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """RMSE, MAE, R², bias, n. Drops pairs where either side is NaN."""
    paired = pd.concat(
        [actual.rename("y"), predicted.rename("p")], axis=1
    ).dropna()
    if paired.empty:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "bias": float("nan"),
        }
    y = paired["y"].to_numpy()
    p = paired["p"].to_numpy()
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": int(len(paired)), "rmse": rmse, "mae": mae, "r2": r2, "bias": bias}


def per_horizon_metrics(
    eval_df: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Take a walk-forward eval DataFrame with pred_rv_<H>/actual_rv_<H> cols,
    return per-horizon metrics summary (one row per horizon)."""
    rows = []
    for h in horizons:
        m = regression_metrics(eval_df[f"actual_rv_{h}"], eval_df[f"pred_rv_{h}"])
        m["horizon"] = h
        rows.append(m)
    return pd.DataFrame(rows).set_index("horizon")[["n", "rmse", "mae", "r2", "bias"]]
