"""Walk-forward tests for the h=1 deviation evaluator (no shuffle, no leakage)."""
import sys

import numpy as np
import pandas as pd

from model.evaluation import regression_metrics
from model.har_model import H1_HAR_FEATURES, HARRVPredictor
from model.lightgbm_model import LightGBMVolPredictor
from model.training import DEFAULT_LGBM_HYPERPARAMS, walk_forward_evaluate_h1

TRAIN_WINDOW = 60
REFIT_EVERY = 5
N = 140


def _bars_from_lv(lv: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Bars whose single-day Garman-Klass vol is exactly exp(lv): O=C so the
    close-open term vanishes, and ln(H/L) = gk_vol * sqrt(2)."""
    v = np.exp(lv)
    x = v * np.sqrt(2.0)
    close = np.full(len(lv), 100.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * np.exp(x / 2.0),
            "low": close * np.exp(-x / 2.0),
            "close": close,
            "volume": np.full(len(lv), 10_000),
        },
        index=idx,
    )


def _har_features_from_lv(lv: pd.Series) -> pd.DataFrame:
    b = lv.rolling(63, min_periods=40).mean()
    return pd.DataFrame(
        {
            "dev_gk": lv - b,
            "har_dev_5": lv.rolling(5).mean() - b,
            "har_dev_22": lv.rolling(22).mean() - b,
        },
        index=lv.index,
    )


def _make_panel(predictable: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=N, freq="B")
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    feature_frames: dict[str, pd.DataFrame] = {}
    for sym in ("AAA", "BBB", "CCC"):
        if predictable:
            # AR(1) in log vol: tomorrow's deviation is forecastable from today's
            lv = np.empty(N)
            lv[0] = -4.5
            for t in range(1, N):
                lv[t] = -0.45 + 0.9 * lv[t - 1] + rng.normal(0, 0.15)
        else:
            # iid log vol: nothing beyond the baseline is forecastable
            lv = rng.normal(-4.5, 0.4, N)
        lv_s = pd.Series(lv, index=idx)
        bars_by_symbol[sym] = _bars_from_lv(lv, idx)
        if predictable:
            feature_frames[sym] = _har_features_from_lv(lv_s)
        else:
            # informationless but row-unique features: a leaking evaluator
            # could memorize them, a correct one scores ~0 OOS
            feature_frames[sym] = pd.DataFrame(
                {c: rng.normal(0, 1, N) for c in H1_HAR_FEATURES}, index=idx
            )
    feature_df = pd.concat(feature_frames, names=["symbol", "date"])
    return feature_df, bars_by_symbol


def test_predictable_signal_yields_positive_r2():
    feature_df, bars = _make_panel(predictable=True, seed=1)
    out = walk_forward_evaluate_h1(
        feature_df, bars,
        model_factory=HARRVPredictor,
        train_window_days=TRAIN_WINDOW,
        refit_every=REFIT_EVERY,
    )
    r2 = regression_metrics(out["actual_dev"], out["predicted_dev"])["r2"]
    assert r2 > 0.3, f"expected OOS r2 > 0.3 on AR(1) signal, got {r2:.4f}"
    print(f"predictable: HAR OOS deviation r2 = {r2:.4f}")


def test_noise_target_scores_near_zero():
    """Leakage guard: with iid log vol and informationless features, a correct
    walk-forward scores ~0 OOS. An evaluator that leaks OOS rows into training
    would let LightGBM memorize the row-unique features and score >> 0."""
    feature_df, bars = _make_panel(predictable=False, seed=2)
    out = walk_forward_evaluate_h1(
        feature_df, bars,
        model_factory=lambda: LightGBMVolPredictor(
            horizon=1, hyperparams=DEFAULT_LGBM_HYPERPARAMS
        ),
        train_window_days=TRAIN_WINDOW,
        refit_every=REFIT_EVERY,
    )
    r2 = regression_metrics(out["actual_dev"], out["predicted_dev"])["r2"]
    assert r2 < 0.1, f"noise target should score ~0 OOS, got {r2:.4f} (leakage?)"
    print(f"leakage_guard: LGBM OOS r2 on noise = {r2:.4f}")


def test_output_alignment_and_oos_only():
    feature_df, bars = _make_panel(predictable=True, seed=3)
    out = walk_forward_evaluate_h1(
        feature_df, bars,
        model_factory=HARRVPredictor,
        train_window_days=TRAIN_WINDOW,
        refit_every=REFIT_EVERY,
    )
    assert list(out.index.names) == ["symbol", "date"]
    np.testing.assert_allclose(
        out["actual_lv"].to_numpy(),
        (out["actual_dev"] + out["baseline_b"]).to_numpy(),
    )
    # predictions must start strictly after the first training window: target
    # rows begin once b_t exists (40 obs), so the OOS region starts at unique
    # target date number TRAIN_WINDOW — well past the baseline warm-up
    oos_dates = np.sort(out.index.get_level_values("date").unique())
    first_possible = feature_df.index.get_level_values("date").unique()[39]
    assert oos_dates[0] > first_possible
    # every (symbol, date) appears at most once
    assert not out.index.duplicated().any()
    print(f"alignment: {len(out)} OOS rows, actual_lv = dev + baseline verified")


def main() -> int:
    test_predictable_signal_yields_positive_r2()
    test_noise_target_scores_near_zero()
    test_output_alignment_and_oos_only()
    print("all walk_forward_h1 tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
