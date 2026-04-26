import sys

import numpy as np
import pandas as pd

from features.technical_indicators import (
    atr,
    bollinger_width,
    intraday_range,
    macd_histogram,
    rsi,
    volume_ratio,
)


def test_constant_prices():
    n = 100
    close = pd.Series([100.0] * n)
    high = pd.Series([100.0] * n)
    low = pd.Series([100.0] * n)
    volume = pd.Series([1000.0] * n)

    bw = bollinger_width(close).dropna()
    assert (bw == 0).all(), "BB width on constants must be 0"

    a = atr(high, low, close).dropna()
    assert (a == 0).all(), "ATR on constants must be 0"

    ir = intraday_range(high, low, close)
    assert (ir == 0).all(), "Intraday range on constants must be 0"

    vr = volume_ratio(volume).dropna()
    np.testing.assert_allclose(vr.values, 1.0)

    # RSI on constants is undefined (gain=loss=0); we replace 0-loss with NaN, so RS=NaN, RSI=NaN
    r = rsi(close)
    assert r.dropna().empty, "RSI on perfectly constant prices should be all NaN"

    print("constant prices: BB, ATR, intraday_range, volume_ratio, RSI all OK")


def test_macd_zero_on_flat_then_responds_to_jump():
    # First half flat, then jump up
    n = 200
    close = pd.Series([100.0] * (n // 2) + [120.0] * (n // 2))
    hist = macd_histogram(close)
    # Pre-jump portion (after MACD warm-up): histogram near 0
    assert hist.iloc[40:99].abs().max() < 1e-6, "MACD should be ~0 on flat prices"
    # After jump: histogram non-zero
    assert hist.iloc[110:130].abs().max() > 0.5, "MACD should react to a price jump"
    print("macd_histogram: flat → ~0, jump → responsive")


def test_rsi_upward_trend_above_70():
    # Strictly increasing series → RSI saturates near 100
    close = pd.Series(np.linspace(100, 200, 100))
    r = rsi(close).dropna()
    # After warm-up, RSI on a monotonically increasing series should be 100 (no losses)
    assert r.iloc[-1] >= 95, f"RSI on monotone increase should be ≥95, got {r.iloc[-1]:.2f}"
    print(f"rsi: monotone increase → RSI[-1] = {r.iloc[-1]:.2f}")


def test_bollinger_width_grows_with_volatility():
    np.random.seed(0)
    close_lo = pd.Series(100 + np.random.normal(0, 0.5, 200).cumsum() * 0.01)
    close_hi = pd.Series(100 + np.random.normal(0, 5.0, 200).cumsum() * 0.01)
    bw_lo = bollinger_width(close_lo).dropna().mean()
    bw_hi = bollinger_width(close_hi).dropna().mean()
    assert bw_hi > bw_lo * 5, f"high-vol BB width should dominate: hi={bw_hi:.4f} lo={bw_lo:.4f}"
    print(f"bollinger_width scales with vol: hi={bw_hi:.4f} >> lo={bw_lo:.4f}")


def main() -> int:
    test_constant_prices()
    test_macd_zero_on_flat_then_responds_to_jump()
    test_rsi_upward_trend_above_70()
    test_bollinger_width_grows_with_volatility()
    print("all technical_indicators tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
