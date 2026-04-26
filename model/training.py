from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from model.evaluation import date_based_ts_split
from model.xgboost_model import XGBoostVolPredictor

logger = logging.getLogger(__name__)


HYPERPARAM_SPACE: dict[str, list] = {
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 5, 10, 20],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 500],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


DEFAULT_HYPERPARAMS: dict = {
    "max_depth": 4,
    "min_child_weight": 5,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def build_training_matrix(
    feature_df: pd.DataFrame,
    returns_by_symbol: dict[str, pd.Series],
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for `horizon`. y[(symbol, t)] = std(returns[symbol].iloc[t+1:t+H+1]).
    Drops rows where y is NaN. Feature NaNs preserved (XGBoost handles natively)."""
    target_name = f"target_rv_{horizon}"
    targets: dict[tuple[str, pd.Timestamp], float] = {}
    for symbol, returns in returns_by_symbol.items():
        idx = returns.index
        n = len(returns)
        for i in range(n - horizon):
            window = returns.iloc[i + 1 : i + 1 + horizon]
            targets[(symbol, idx[i])] = float(window.std(ddof=0))

    if not targets:
        return feature_df.iloc[:0].copy(), pd.Series(dtype=float, name=target_name)

    y = pd.Series(targets, name=target_name)
    y.index.names = ["symbol", "date"]

    aligned = feature_df.join(y, how="inner").dropna(subset=[target_name])
    X = aligned.drop(columns=[target_name])
    y_aligned = aligned[target_name]
    return X, y_aligned


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[dict, float]:
    """Random search over HYPERPARAM_SPACE, scored as mean OOS R² across
    date-based time-series CV folds. Returns (best_params, best_mean_r2)."""
    rng = np.random.default_rng(seed)
    dates = pd.Series(X.index.get_level_values("date"))
    splits = list(date_based_ts_split(dates, n_splits=n_splits))

    best_score = -np.inf
    best_params: dict | None = None

    for trial in range(n_trials):
        params = {k: rng.choice(v).item() for k, v in HYPERPARAM_SPACE.items()}
        fold_scores = []
        for train_idx, test_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=0,
                **params,
            )
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_te)
            fold_scores.append(_r2_score(y_te.to_numpy(), preds))
        if not fold_scores:
            continue
        mean_score = float(np.nanmean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            logger.debug("trial %d: r2=%.4f params=%s", trial, mean_score, params)

    if best_params is None:
        raise RuntimeError("hyperparameter tuning produced no valid results")
    logger.info("best hyperparams (mean CV R²=%.4f): %s", best_score, best_params)
    return best_params, best_score


def walk_forward_evaluate_xgboost(
    feature_df: pd.DataFrame,
    returns_by_symbol: dict[str, pd.Series],
    horizon: int,
    train_window_days: int = 504,
    refit_every: int = 21,
    hyperparams: dict | None = None,
    artifact_dir: Path | None = None,
    importance_log_path: Path | None = None,
) -> pd.DataFrame:
    """Walk forward through dates. Every `refit_every` trading days:
    - Slice training window: rows whose date is in the most recent `train_window_days` dates ending at t.
    - Fit XGBoost (using passed hyperparams; if None, tune once on the first window).
    - Predict for the next `refit_every` dates (OOS).
    - Optionally save model artifact + feature-importance row.

    Returns DataFrame indexed by (symbol, date) with columns: predicted, actual."""
    X, y = build_training_matrix(feature_df, returns_by_symbol, horizon)

    dates = pd.Series(X.index.get_level_values("date"))
    unique_dates = pd.Index(np.sort(dates.unique()))
    n_dates = len(unique_dates)

    if n_dates <= train_window_days + refit_every:
        raise ValueError(
            f"need >{train_window_days + refit_every} dates, got {n_dates}"
        )

    if hyperparams is None:
        first_train_dates = set(unique_dates[:train_window_days])
        tune_mask = dates.isin(first_train_dates).to_numpy()
        X_tune = X.iloc[tune_mask]
        y_tune = y.iloc[tune_mask]
        hyperparams, _ = tune_hyperparameters(X_tune, y_tune)

    importance_rows: list[dict] = []
    results: list[pd.DataFrame] = []

    for i in range(train_window_days, n_dates, refit_every):
        train_dates = set(unique_dates[i - train_window_days : i])
        oos_end = min(i + refit_every, n_dates)
        oos_dates = set(unique_dates[i:oos_end])

        train_mask = dates.isin(train_dates).to_numpy()
        oos_mask = dates.isin(oos_dates).to_numpy()

        X_tr, y_tr = X.iloc[train_mask], y.iloc[train_mask]
        X_oos, y_oos = X.iloc[oos_mask], y.iloc[oos_mask]

        if len(X_tr) == 0 or len(X_oos) == 0:
            continue

        predictor = XGBoostVolPredictor(horizon=horizon, hyperparams=hyperparams)
        predictor.fit(X_tr, y_tr)
        preds = predictor.predict(X_oos)

        results.append(pd.DataFrame(
            {"predicted": preds, "actual": y_oos.to_numpy()},
            index=y_oos.index,
        ))

        train_end_date = max(train_dates)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            predictor.save(artifact_dir / f"xgb_h{horizon}_{pd.Timestamp(train_end_date).date().isoformat()}.joblib")

        imp = predictor.feature_importance()
        row = {"refit_date": train_end_date, **imp.to_dict()}
        importance_rows.append(row)

    if importance_log_path is not None and importance_rows:
        importance_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(importance_rows).to_csv(importance_log_path, index=False)

    if not results:
        return pd.DataFrame(columns=["predicted", "actual"])
    return pd.concat(results)
