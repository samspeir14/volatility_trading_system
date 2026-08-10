from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from features.target import build_h1_deviation_target
from model.evaluation import date_based_ts_split
from model.lightgbm_model import LightGBMVolPredictor
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


LGBM_HYPERPARAM_SPACE: dict[str, list] = {
    "max_depth": [3, 4, 5, 6, -1],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 500],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [5, 10, 20],
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


DEFAULT_LGBM_HYPERPARAMS: dict = {
    "max_depth": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
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
    space: dict[str, list] | None = None,
    model_factory: Callable[[dict], object] | None = None,
) -> tuple[dict, float]:
    """Random search scored as mean OOS R² across date-based TS CV folds.

    Defaults to the XGBoost search space + XGBRegressor for backwards
    compatibility. Pass `space=LGBM_HYPERPARAM_SPACE` and a LightGBM
    `model_factory` to tune LightGBM with the same logic.
    """
    if space is None:
        space = HYPERPARAM_SPACE
    if model_factory is None:
        model_factory = lambda params: xgb.XGBRegressor(
            objective="reg:squarederror", random_state=0, **params,
        )

    rng = np.random.default_rng(seed)
    dates = pd.Series(X.index.get_level_values("date"))
    splits = list(date_based_ts_split(dates, n_splits=n_splits))

    best_score = -np.inf
    best_params: dict | None = None

    for trial in range(n_trials):
        params = {k: rng.choice(v).item() for k, v in space.items()}
        fold_scores = []
        for train_idx, test_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            try:
                model = model_factory(params)
                # XGBoost accepts verbose=, LightGBM doesn't on .fit()
                try:
                    model.fit(X_tr, y_tr, verbose=False)
                except TypeError:
                    model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
            except Exception as exc:
                logger.warning("trial %d fold failed: %s", trial, exc)
                continue
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


def _lgbm_factory(params: dict):
    """LightGBM regressor factory for tune_hyperparameters."""
    full = dict(params)
    if full.get("subsample", 1.0) < 1.0 and "bagging_freq" not in full:
        full["bagging_freq"] = 1
    import lightgbm as lgb_lib
    return lgb_lib.LGBMRegressor(
        objective="regression", random_state=0, verbose=-1, **full,
    )


H1_TARGET_NAME = "target_dev_h1"
H1_BASELINE_NAME = "baseline_b"


def build_h1_training_matrix(
    feature_df: pd.DataFrame,
    bars_by_symbol: dict[str, pd.DataFrame],
    feature_subset: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y, b) for the h=1 deviation model, pooled across tickers.
    y[(s,t)] = log(gk_vol_{t+1}+eps) - b_t (see features.target); b is the
    aligned baseline b_t (finite wherever y is, since y requires it).
    Drops rows where y is NaN. Feature NaNs preserved (LightGBM handles
    natively; HAR drops them at fit)."""
    y, b, _lv = build_h1_deviation_target(bars_by_symbol)

    if y.empty:
        cols = feature_subset if feature_subset is not None else list(feature_df.columns)
        return (
            feature_df.iloc[:0].reindex(columns=cols),
            pd.Series(dtype=float, name=H1_TARGET_NAME),
            pd.Series(dtype=float, name=H1_BASELINE_NAME),
        )

    aligned = feature_df.join(y.rename(H1_TARGET_NAME), how="inner")
    aligned = aligned.join(b.rename(H1_BASELINE_NAME), how="left")
    aligned = aligned.dropna(subset=[H1_TARGET_NAME])

    cols = feature_subset if feature_subset is not None else list(feature_df.columns)
    X = aligned[cols]
    return X, aligned[H1_TARGET_NAME], aligned[H1_BASELINE_NAME]


def tune_h1_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    train_window_days: int = 504,
    n_trials: int = 50,
) -> dict:
    """Tune LightGBM on the FIRST training window only (no lookahead into the
    walk-forward OOS region). Slices the earliest `train_window_days` unique
    dates from (X, y) and runs the shared random search."""
    dates = pd.Series(X.index.get_level_values("date"))
    unique_dates = pd.Index(np.sort(dates.unique()))
    first_train_dates = set(unique_dates[:train_window_days])
    mask = dates.isin(first_train_dates).to_numpy()
    params, _ = tune_hyperparameters(
        X.iloc[mask], y.iloc[mask],
        n_trials=n_trials,
        space=LGBM_HYPERPARAM_SPACE,
        model_factory=_lgbm_factory,
    )
    return params


def walk_forward_evaluate_h1(
    feature_df: pd.DataFrame,
    bars_by_symbol: dict[str, pd.DataFrame],
    model_factory: Callable[[], object],
    feature_subset: list[str] | None = None,
    train_window_days: int = 504,
    refit_every: int = 21,
    artifact_dir: Path | None = None,
    artifact_prefix: str | None = None,
    importance_log_path: Path | None = None,
    importance_acc: list[pd.Series] | None = None,
) -> pd.DataFrame:
    """Walk-forward evaluation on the h=1 deviation target. Same window
    semantics as walk_forward_evaluate_lightgbm: strictly time-ordered, no
    shuffle — train on the trailing `train_window_days` unique dates, predict
    the next `refit_every` dates OOS, roll forward.

    `model_factory` returns a fresh fitted-API object (fit/predict, optional
    feature_importance/save) per refit — LightGBMVolPredictor, HARRVPredictor,
    or anything matching that protocol.

    Returns a (symbol, date)-indexed DataFrame with columns:
    predicted_dev, actual_dev, baseline_b, actual_lv (= actual_dev + baseline_b).
    """
    X, y, b = build_h1_training_matrix(feature_df, bars_by_symbol, feature_subset)

    dates = pd.Series(X.index.get_level_values("date"))
    unique_dates = pd.Index(np.sort(dates.unique()))
    n_dates = len(unique_dates)

    if n_dates <= train_window_days + refit_every:
        raise ValueError(
            f"need >{train_window_days + refit_every} dates, got {n_dates}"
        )

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
        b_oos = b.iloc[oos_mask]

        if len(X_tr) == 0 or len(X_oos) == 0:
            continue

        predictor = model_factory()
        predictor.fit(X_tr, y_tr)
        preds = predictor.predict(X_oos)

        results.append(pd.DataFrame(
            {
                "predicted_dev": preds,
                "actual_dev": y_oos.to_numpy(),
                "baseline_b": b_oos.to_numpy(),
            },
            index=y_oos.index,
        ))

        train_end_date = max(train_dates)
        if artifact_dir is not None and artifact_prefix is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            predictor.save(
                artifact_dir
                / f"{artifact_prefix}_{pd.Timestamp(train_end_date).date().isoformat()}.joblib"
            )

        if hasattr(predictor, "feature_importance"):
            imp = predictor.feature_importance()
            importance_rows.append({"refit_date": train_end_date, **imp.to_dict()})
            if importance_acc is not None:
                importance_acc.append(imp)

    if importance_log_path is not None and importance_rows:
        importance_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(importance_rows).to_csv(importance_log_path, index=False)

    if not results:
        return pd.DataFrame(
            columns=["predicted_dev", "actual_dev", "baseline_b", "actual_lv"]
        )
    out = pd.concat(results)
    out["actual_lv"] = out["actual_dev"] + out["baseline_b"]
    return out


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


def walk_forward_evaluate_lightgbm(
    feature_df: pd.DataFrame,
    returns_by_symbol: dict[str, pd.Series],
    horizon: int,
    train_window_days: int = 504,
    refit_every: int = 21,
    hyperparams: dict | None = None,
    artifact_dir: Path | None = None,
    importance_log_path: Path | None = None,
) -> pd.DataFrame:
    """LightGBM equivalent of walk_forward_evaluate_xgboost. Same window
    semantics, same target construction. Saves artifacts as
    `lgbm_h{H}_{end_date}.joblib`. If hyperparams=None, tunes once on the
    first training window using LGBM_HYPERPARAM_SPACE."""
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
        hyperparams, _ = tune_hyperparameters(
            X_tune, y_tune,
            space=LGBM_HYPERPARAM_SPACE,
            model_factory=_lgbm_factory,
        )

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

        predictor = LightGBMVolPredictor(horizon=horizon, hyperparams=hyperparams)
        predictor.fit(X_tr, y_tr)
        preds = predictor.predict(X_oos)

        results.append(pd.DataFrame(
            {"predicted": preds, "actual": y_oos.to_numpy()},
            index=y_oos.index,
        ))

        train_end_date = max(train_dates)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            predictor.save(artifact_dir / f"lgbm_h{horizon}_{pd.Timestamp(train_end_date).date().isoformat()}.joblib")

        imp = predictor.feature_importance()
        row = {"refit_date": train_end_date, **imp.to_dict()}
        importance_rows.append(row)

    if importance_log_path is not None and importance_rows:
        importance_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(importance_rows).to_csv(importance_log_path, index=False)

    if not results:
        return pd.DataFrame(columns=["predicted", "actual"])
    return pd.concat(results)
