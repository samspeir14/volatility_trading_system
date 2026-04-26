import sys

import numpy as np
import pandas as pd

from model.evaluation import regression_metrics
from model.training import (
    DEFAULT_HYPERPARAMS,
    build_training_matrix,
    walk_forward_evaluate_xgboost,
)


def _make_synthetic_panel(
    symbols: list[str], n_days: int, horizon: int, seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Synthetic panel where forward RV depends on a noisy linear combination of features."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days, freq="B")

    feat_frames: dict[str, pd.DataFrame] = {}
    returns_by_symbol: dict[str, pd.Series] = {}

    for sym in symbols:
        # Drive returns from a latent vol that's a function of features
        f1 = rng.normal(0, 1, n_days).cumsum() / 50  # slow random walk
        f2 = rng.normal(0, 1, n_days)
        latent_vol = 0.005 + 0.005 * np.abs(f1) + 0.002 * np.abs(f2)
        z = rng.standard_normal(n_days)
        returns = latent_vol * z
        feat_frames[sym] = pd.DataFrame(
            {"f1": f1, "f2": f2, "noise": rng.normal(0, 1, n_days)},
            index=idx,
        )
        returns_by_symbol[sym] = pd.Series(returns, index=idx)
    feature_df = pd.concat(feat_frames, names=["symbol", "date"])
    return feature_df, returns_by_symbol


def test_walk_forward_shape_and_dates():
    horizon = 5
    train_window = 100
    refit_every = 21
    n_days = 350
    symbols = ["A", "B", "C"]
    feature_df, returns = _make_synthetic_panel(symbols, n_days, horizon, seed=1)

    out = walk_forward_evaluate_xgboost(
        feature_df, returns, horizon=horizon,
        train_window_days=train_window, refit_every=refit_every,
        hyperparams=DEFAULT_HYPERPARAMS,
    )

    assert list(out.columns) == ["predicted", "actual"]
    assert out.index.names == ["symbol", "date"]
    # All OOS dates should be after the first training window ends
    out_dates = out.index.get_level_values("date").unique()
    # Build matrix gives n_days - horizon dates per symbol; first OOS date is at position train_window
    panel_dates = build_training_matrix(feature_df, returns, horizon)[0].index.get_level_values("date").unique()
    panel_dates = pd.DatetimeIndex(np.sort(panel_dates))
    expected_first_oos = panel_dates[train_window]
    assert out_dates.min() >= expected_first_oos, "OOS dates leak into training window"
    print(f"shape_and_dates: {len(out)} OOS rows, dates start at {out_dates.min().date()}")


def test_leakage_guard_shuffled_target():
    """If we shuffle the target, OOS R² should collapse to ≤ 0 (no real signal).
    With true target it should be > 0."""
    horizon = 5
    train_window = 150
    refit_every = 21
    n_days = 400
    symbols = ["A", "B", "C"]
    feature_df, returns = _make_synthetic_panel(symbols, n_days, horizon, seed=7)

    # 1) True targets: should produce positive R²
    out_true = walk_forward_evaluate_xgboost(
        feature_df, returns, horizon=horizon,
        train_window_days=train_window, refit_every=refit_every,
        hyperparams=DEFAULT_HYPERPARAMS,
    )
    r2_true = regression_metrics(out_true["actual"], out_true["predicted"])["r2"]
    print(f"  true target: R² = {r2_true:.4f} (n={len(out_true)})")

    # 2) Shuffled per-symbol returns → predicted vol shouldn't track actual
    rng = np.random.default_rng(123)
    shuffled_returns = {
        sym: pd.Series(rng.permutation(returns[sym].values), index=returns[sym].index)
        for sym in symbols
    }
    out_sh = walk_forward_evaluate_xgboost(
        feature_df, shuffled_returns, horizon=horizon,
        train_window_days=train_window, refit_every=refit_every,
        hyperparams=DEFAULT_HYPERPARAMS,
    )
    r2_sh = regression_metrics(out_sh["actual"], out_sh["predicted"])["r2"]
    print(f"  shuffled target: R² = {r2_sh:.4f} (n={len(out_sh)})")

    assert r2_true > r2_sh, (
        f"true R² ({r2_true:.4f}) should exceed shuffled R² ({r2_sh:.4f})"
    )
    print(f"leakage_guard: true R² > shuffled R² ✓")


def test_minimum_refits_in_oos_window():
    horizon = 5
    train_window = 150
    refit_every = 21
    n_days = 400
    feature_df, returns = _make_synthetic_panel(["A", "B"], n_days, horizon, seed=2)
    out = walk_forward_evaluate_xgboost(
        feature_df, returns, horizon=horizon,
        train_window_days=train_window, refit_every=refit_every,
        hyperparams=DEFAULT_HYPERPARAMS,
    )
    # OOS window ≈ n_days - horizon - train_window. Refits ≈ that / refit_every
    n_unique_dates = out.index.get_level_values("date").nunique()
    n_refits = (n_unique_dates + refit_every - 1) // refit_every
    assert n_refits >= 5, f"only {n_refits} refits in OOS window"
    print(f"minimum_refits: ~{n_refits} refits across {n_unique_dates} OOS dates")


def main() -> int:
    test_walk_forward_shape_and_dates()
    test_leakage_guard_shuffled_target()
    test_minimum_refits_in_oos_window()
    print("all walk_forward_xgboost tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
