from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from model.evaluation import date_based_ts_split
from model.training import build_training_matrix

from experiments.predictors import VolPredictorProtocol

logger = logging.getLogger(__name__)


PredictorFactory = Callable[[int, dict], VolPredictorProtocol]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def first_window_slice(
    X: pd.DataFrame,
    y: pd.Series,
    train_window_days: int = 504,
) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.Series(X.index.get_level_values("date"))
    unique_dates = pd.Index(np.sort(dates.unique()))
    if len(unique_dates) < train_window_days:
        return X, y
    first_dates = set(unique_dates[:train_window_days])
    mask = dates.isin(first_dates).to_numpy()
    return X.iloc[mask], y.iloc[mask]


def tune_generic(
    X: pd.DataFrame,
    y: pd.Series,
    predictor_factory: PredictorFactory,
    horizon: int,
    space: dict[str, list],
    n_trials: int = 50,
    n_splits: int = 3,
    seed: int = 0,
) -> tuple[dict, float]:
    """Random search over `space`, scored as mean OOS R² across date-based TS CV folds.
    Mirrors model.training.tune_hyperparameters but parameterized on a predictor factory."""
    rng = np.random.default_rng(seed)
    dates = pd.Series(X.index.get_level_values("date"))
    splits = list(date_based_ts_split(dates, n_splits=n_splits))

    best_score = -np.inf
    best_params: dict | None = None

    for trial in range(n_trials):
        params = {k: rng.choice(v).item() for k, v in space.items()}
        fold_scores: list[float] = []
        for train_idx, test_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            try:
                predictor = predictor_factory(horizon, params)
                predictor.fit(X_tr, y_tr)
                preds = predictor.predict(X_te)
            except Exception as exc:
                logger.warning("trial %d fold failed: %s", trial, exc)
                continue
            fold_scores.append(_r2(y_te.to_numpy(), preds))
        if not fold_scores:
            continue
        mean_score = float(np.nanmean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        raise RuntimeError("tune_generic produced no valid results")
    return best_params, best_score


def walk_forward_generic(
    feature_df: pd.DataFrame,
    returns_by_symbol: dict[str, pd.Series],
    horizon: int,
    predictor_factory: PredictorFactory,
    hyperparams: dict,
    train_window_days: int = 504,
    refit_every: int = 21,
    importance_acc: list[pd.Series] | None = None,
) -> pd.DataFrame:
    """Generic walk-forward evaluator. Mirrors model.training.walk_forward_evaluate_xgboost
    but accepts any predictor that implements VolPredictorProtocol.

    Reuses model.training.build_training_matrix for target construction so results are
    directly comparable to the live XGBoost baseline."""
    X, y = build_training_matrix(feature_df, returns_by_symbol, horizon)

    dates = pd.Series(X.index.get_level_values("date"))
    unique_dates = pd.Index(np.sort(dates.unique()))
    n_dates = len(unique_dates)

    if n_dates <= train_window_days + refit_every:
        raise ValueError(
            f"need >{train_window_days + refit_every} dates, got {n_dates}"
        )

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

        predictor = predictor_factory(horizon, hyperparams)
        predictor.fit(X_tr, y_tr)
        preds = predictor.predict(X_oos)
        results.append(pd.DataFrame(
            {"predicted": preds, "actual": y_oos.to_numpy()},
            index=y_oos.index,
        ))

        if importance_acc is not None:
            try:
                importance_acc.append(predictor.feature_importance())
            except Exception as exc:
                logger.warning("feature_importance failed at refit %d: %s", i, exc)

    if not results:
        return pd.DataFrame(columns=["predicted", "actual"])
    return pd.concat(results)


def pick_top_features_by_mean_rank(
    importance_acc: list[pd.Series],
    top_n: int = 20,
) -> list[str]:
    """Aggregate per-refit importance Series into a feature list of size `top_n` by
    mean rank (lowest mean rank = most important across refits). Robust to scale
    drift across refits where a single huge-gain refit could dominate raw means."""
    if not importance_acc:
        return []
    rank_frame = pd.concat(
        [s.rank(ascending=False, method="average") for s in importance_acc],
        axis=1,
    )
    mean_rank = rank_frame.mean(axis=1).sort_values(ascending=True)
    return mean_rank.head(top_n).index.tolist()


def ridge_stack(
    pred_frames: dict[str, pd.DataFrame],
    n_splits: int = 5,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Train a Ridge meta-learner over base-learner OOS predictions using
    date-based time-series CV. Returns OOS stacker predictions in the same
    (symbol, date) MultiIndex shape as the inputs.

    Note: this is slightly optimistic vs. a fully nested setup because the
    base-learner OOS predictions in early dates were trained on a smaller
    window than they would have been in proper nesting. For experimentation
    this is an acceptable tradeoff."""
    if not pred_frames:
        return pd.DataFrame(columns=["predicted", "actual"])

    names = list(pred_frames.keys())
    aligned = pd.DataFrame({
        f"pred_{name}": pred_frames[name]["predicted"]
        for name in names
    })
    # Take actual from any frame (they should all match after inner-join)
    aligned["actual"] = pred_frames[names[0]]["actual"]
    aligned = aligned.dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["predicted", "actual"])

    feature_cols = [f"pred_{n}" for n in names]
    dates = pd.Series(aligned.index.get_level_values("date"))
    splits = list(date_based_ts_split(dates, n_splits=n_splits))

    oos_pieces: list[pd.DataFrame] = []
    for train_idx, test_idx in splits:
        X_tr = aligned.iloc[train_idx][feature_cols].to_numpy()
        y_tr = aligned.iloc[train_idx]["actual"].to_numpy()
        X_te = aligned.iloc[test_idx][feature_cols].to_numpy()
        y_te = aligned.iloc[test_idx]["actual"].to_numpy()
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        oos_pieces.append(pd.DataFrame(
            {"predicted": preds, "actual": y_te},
            index=aligned.iloc[test_idx].index,
        ))

    if not oos_pieces:
        return pd.DataFrame(columns=["predicted", "actual"])
    return pd.concat(oos_pieces)
