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


def within_ticker_r2(actual: pd.Series, predicted: pd.Series, level: str = "symbol") -> float:
    """R² with SS_total computed within-symbol: 1 − SSE / Σ(y − ȳ_symbol)².

    Pooled R² across tickers gets credit for knowing that high-vol names run
    hotter than low-vol names — a level difference the options market already
    prices into IV. This variant scores skill relative to a per-symbol-mean
    forecast, so a model that only knows each ticker's vol level scores ~0.
    Requires a MultiIndex with a `level` (default "symbol") on both series.
    Drops pairs where either side is NaN."""
    paired = pd.concat(
        [actual.rename("y"), predicted.rename("p")], axis=1
    ).dropna()
    if paired.empty:
        return float("nan")
    y = paired["y"]
    ss_res = float(((paired["p"] - y) ** 2).sum())
    ss_within = float(((y - y.groupby(level=level).transform("mean")) ** 2).sum())
    if ss_within <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_within


def qlike(actual_var: pd.Series, forecast_var: pd.Series) -> float:
    """QLIKE loss on variances: mean( a/f − log(a/f) − 1 ). Zero when the
    forecast is perfect; penalizes under-forecasting variance more than
    over-forecasting, which matches the economics of short-vol positions.
    Inputs are DAILY VARIANCES (vol²). Rows where either side is NaN or
    nonpositive are dropped (log requires strictly positive values)."""
    paired = pd.concat(
        [actual_var.rename("a"), forecast_var.rename("f")], axis=1
    ).dropna()
    paired = paired[(paired["a"] > 0) & (paired["f"] > 0)]
    if paired.empty:
        return float("nan")
    ratio = paired["a"].to_numpy() / paired["f"].to_numpy()
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def per_ticker_r2_median(
    actual: pd.Series, predicted: pd.Series, level: str = "symbol"
) -> float:
    """Median across symbols of the per-symbol R². Complements the pooled
    deviation R²: a pooled number can be carried by a few high-variance names,
    the median cannot. Requires a MultiIndex with `level` on both series;
    symbols with <3 paired rows or zero within-symbol variance are skipped."""
    paired = pd.concat(
        [actual.rename("y"), predicted.rename("p")], axis=1
    ).dropna()
    if paired.empty:
        return float("nan")
    r2s: list[float] = []
    for _, grp in paired.groupby(level=level):
        if len(grp) < 3:
            continue
        y = grp["y"].to_numpy()
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            continue
        ss_res = float(np.sum((grp["p"].to_numpy() - y) ** 2))
        r2s.append(1.0 - ss_res / ss_tot)
    if not r2s:
        return float("nan")
    return float(np.median(r2s))


def sign_hit_rate(actual: pd.Series, predicted: pd.Series) -> float:
    """Fraction of rows where the forecast gets the deviation's SIGN right.
    The cost gate monetizes direction-plus-threshold, not squared error, so
    this is the report-only trade-aligned complement to within-ticker R².
    Rows with zero actual (no direction to call) or NaN on either side are
    dropped."""
    paired = pd.concat(
        [actual.rename("y"), predicted.rename("p")], axis=1
    ).dropna()
    paired = paired[paired["y"] != 0.0]
    if paired.empty:
        return float("nan")
    hits = np.sign(paired["y"].to_numpy()) == np.sign(paired["p"].to_numpy())
    return float(hits.mean())


def per_ticker_spearman_median(
    actual: pd.Series, predicted: pd.Series, level: str = "symbol"
) -> float:
    """Median across symbols of the within-symbol Spearman rank correlation
    between forecast and actual deviation — pure time-ordering skill, immune
    to scale and outliers. Report-only diagnostic. Symbols with <3 paired
    rows are skipped."""
    paired = pd.concat(
        [actual.rename("y"), predicted.rename("p")], axis=1
    ).dropna()
    if paired.empty:
        return float("nan")
    rhos: list[float] = []
    for _, grp in paired.groupby(level=level):
        if len(grp) < 3:
            continue
        rho = grp["y"].rank().corr(grp["p"].rank())
        if np.isfinite(rho):
            rhos.append(float(rho))
    if not rhos:
        return float("nan")
    return float(np.median(rhos))


def r2_vs_baseline(actual: pd.Series, predicted: pd.Series, baseline: pd.Series) -> float:
    """Out-of-sample R² against a competing forecast: 1 − SSE_model / SSE_baseline
    (Campbell–Thompson style). Positive = model beats the baseline; 0 = ties it.
    Rows where any of the three is NaN are dropped, so both forecasts are
    scored on the identical sample."""
    paired = pd.concat(
        [actual.rename("y"), predicted.rename("p"), baseline.rename("b")], axis=1
    ).dropna()
    if paired.empty:
        return float("nan")
    sse_model = float(((paired["p"] - paired["y"]) ** 2).sum())
    sse_base = float(((paired["b"] - paired["y"]) ** 2).sum())
    if sse_base <= 0:
        return float("nan")
    return 1.0 - sse_model / sse_base


def lagged_rv_forecast(
    returns_by_symbol: dict[str, pd.Series],
    horizon: int,
) -> pd.Series:
    """Random-walk vol forecast: predicted forward-`horizon`-day RV at (symbol, t)
    = std(returns[t−horizon+1 : t+1], ddof=0), the trailing window ending at t.
    Same estimator as the training target (std of the *next* horizon days), so
    the two windows are adjacent and non-overlapping. First horizon−1 rows per
    symbol are NaN. Returns a Series indexed by (symbol, date)."""
    parts: list[pd.Series] = []
    for symbol, returns in returns_by_symbol.items():
        trailing = returns.rolling(horizon).std(ddof=0)
        parts.append(
            pd.Series(
                trailing.to_numpy(),
                index=pd.MultiIndex.from_product(
                    [[symbol], trailing.index], names=["symbol", "date"]
                ),
            )
        )
    return pd.concat(parts)


def h1_metrics_from_predictions(preds_long: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute the h=1 report metrics from a long-format predictions frame
    (columns: symbol, date, model, predicted_dev, actual_dev, baseline_b,
    actual_lv). This is the SINGLE definition used both by the retrain job
    (to report) and by scripts/reconcile_h1_metrics.py (to verify the stored
    predictions reproduce the reported numbers within 1e-6)."""
    out: dict[str, dict[str, float]] = {}
    for model_name, grp in preds_long.groupby("model"):
        g = grp.set_index(["symbol", "date"])
        actual_dev = g["actual_dev"]
        pred_dev = g["predicted_dev"]
        actual_var = np.exp(2.0 * g["actual_lv"])
        forecast_var = np.exp(2.0 * (g["baseline_b"] + pred_dev))
        m = regression_metrics(actual_dev, pred_dev)
        out[str(model_name)] = {
            "n": m["n"],
            "dev_r2_pooled": m["r2"],
            "dev_r2_within": within_ticker_r2(actual_dev, pred_dev),
            "dev_r2_ticker_median": per_ticker_r2_median(actual_dev, pred_dev),
            "qlike_level": qlike(actual_var, forecast_var),
            "sign_hit_rate": sign_hit_rate(actual_dev, pred_dev),
            "dev_spearman_median": per_ticker_spearman_median(actual_dev, pred_dev),
        }
    return out


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
