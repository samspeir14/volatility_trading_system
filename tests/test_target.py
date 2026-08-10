"""Tests for the h=1 deviation target (features/target.py).

Includes the spec-required no-lookahead assertion: the baseline b_t must not
use any data after t.
"""
import math
import sys

import numpy as np
import pandas as pd

from features.target import (
    BASELINE_MIN_OBS,
    GK_EPS,
    build_h1_deviation_target,
    daily_ohlc_vol,
    log_vol,
    rolling_log_vol_baseline,
)


def _make_bars(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLC bars with realistic intraday range."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    hi_pad = np.abs(rng.normal(0, 0.008, n))
    lo_pad = np.abs(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * np.exp(hi_pad)
    low = np.minimum(open_, close) * np.exp(-lo_pad)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(1_000, 100_000, n)},
        index=idx,
    )


def test_gk_value_matches_formula():
    bars = _make_bars(10)
    vol = daily_ohlc_vol(bars)
    i = 4
    log_hl = math.log(bars["high"].iloc[i] / bars["low"].iloc[i])
    log_co = math.log(bars["close"].iloc[i] / bars["open"].iloc[i])
    gk = 0.5 * log_hl ** 2 - (2.0 * math.log(2.0) - 1.0) * log_co ** 2
    expected = math.sqrt(max(gk, 0.0))
    assert math.isclose(float(vol.iloc[i]), expected, rel_tol=1e-12)
    print("gk_value: matches hand-computed Garman-Klass")


def test_fallback_parkinson_when_open_bad():
    bars = _make_bars(10)
    bars.iloc[3, bars.columns.get_loc("open")] = 0.0  # bad open → GK undefined
    vol = daily_ohlc_vol(bars)
    log_hl = math.log(bars["high"].iloc[3] / bars["low"].iloc[3])
    expected = math.sqrt(log_hl ** 2 / (4.0 * math.log(2.0)))
    assert math.isclose(float(vol.iloc[3]), expected, rel_tol=1e-12)
    print("fallback_parkinson: bad open falls through to Parkinson")


def test_fallback_c2c_when_range_bad():
    bars = _make_bars(10)
    bars.iloc[3, bars.columns.get_loc("open")] = 0.0
    bars.iloc[3, bars.columns.get_loc("high")] = 0.0
    bars.iloc[3, bars.columns.get_loc("low")] = 0.0
    vol = daily_ohlc_vol(bars)
    expected = abs(math.log(bars["close"].iloc[3] / bars["close"].iloc[2]))
    assert math.isclose(float(vol.iloc[3]), expected, rel_tol=1e-12)
    print("fallback_c2c: bad range falls through to |close-to-close return|")


def test_baseline_no_lookahead():
    """Spec-required: b_t must not use data after t. Perturb everything after
    a cut date; the baseline up to the cut must be bitwise identical."""
    bars = _make_bars(120, seed=1)
    lv = log_vol(daily_ohlc_vol(bars))
    b_full = rolling_log_vol_baseline(lv)

    cut = 80
    lv_perturbed = lv.copy()
    lv_perturbed.iloc[cut + 1 :] = lv_perturbed.iloc[cut + 1 :] * 5.0 + 3.0
    b_perturbed = rolling_log_vol_baseline(lv_perturbed)

    np.testing.assert_array_equal(
        b_full.iloc[: cut + 1].to_numpy(), b_perturbed.iloc[: cut + 1].to_numpy()
    )
    print(f"no_lookahead: b_t identical through t={cut} after perturbing t>{cut}")


def test_baseline_min_obs_boundary():
    bars = _make_bars(60, seed=2)
    b = rolling_log_vol_baseline(log_vol(daily_ohlc_vol(bars)))
    assert np.isnan(b.iloc[BASELINE_MIN_OBS - 2]), "obs 39 should be NaN"
    assert np.isfinite(b.iloc[BASELINE_MIN_OBS - 1]), "obs 40 should be finite"
    print(f"min_obs: NaN at {BASELINE_MIN_OBS - 1} obs, finite at {BASELINE_MIN_OBS}")


def test_target_uses_next_day_vol():
    bars = _make_bars(80, seed=3)
    y, b, lv = build_h1_deviation_target({"AAPL": bars})

    lv_a = lv.loc["AAPL"]
    b_a = b.loc["AAPL"]
    for date in y.loc["AAPL"].index:
        i = bars.index.get_loc(date)
        expected = float(lv_a.iloc[i + 1]) - float(b_a.iloc[i])
        assert math.isclose(float(y.loc[("AAPL", date)]), expected, rel_tol=1e-12)
    # last row has no t+1 → never in y; rows before min_obs have NaN b → not in y
    assert ("AAPL", bars.index[-1]) not in y.index
    assert ("AAPL", bars.index[0]) not in y.index
    print(f"target_shift: {len(y)} rows verified as lv_(t+1) - b_t")


def test_flat_bar_yields_nan_not_zero():
    """Reviewer-flagged: a halted/placeholder bar (O=H=L=C) reads exactly 0
    on GK and Parkinson. It must become NaN, not log(eps)=-18.4 poisoning the
    baseline for 63 sessions."""
    bars = _make_bars(10)
    for col in ("open", "high", "low"):
        bars.iloc[4, bars.columns.get_loc(col)] = bars["close"].iloc[3]
    bars.iloc[4, bars.columns.get_loc("close")] = bars["close"].iloc[3]  # c2c = 0 too
    vol = daily_ohlc_vol(bars)
    assert np.isnan(vol.iloc[4]), f"flat bar should be NaN, got {vol.iloc[4]}"
    assert np.isfinite(vol.iloc[5])
    print("flat_bar: O=H=L=C day drops out as NaN")


def test_log_vol_epsilon_guards_zero():
    zero = pd.Series([0.0, 0.01])
    lv = log_vol(zero)
    assert np.isfinite(lv.iloc[0]) and math.isclose(lv.iloc[0], math.log(GK_EPS))
    print("log_eps: zero vol maps to log(eps), stays finite")


def main() -> int:
    test_gk_value_matches_formula()
    test_fallback_parkinson_when_open_bad()
    test_fallback_c2c_when_range_bad()
    test_baseline_no_lookahead()
    test_baseline_min_obs_boundary()
    test_target_uses_next_day_vol()
    test_flat_bar_yields_nan_not_zero()
    test_log_vol_epsilon_guards_zero()
    print("all target tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
