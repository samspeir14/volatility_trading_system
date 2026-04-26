import math
import sys

import numpy as np
import pandas as pd

from model.training import build_training_matrix


def _make_panel(symbols: list[str], n: int, seed: int = 0) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build a synthetic feature DataFrame (MultiIndex) and per-symbol returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    feature_frames = {}
    returns = {}
    for sym in symbols:
        feature_frames[sym] = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_b": rng.normal(0, 1, n),
            },
            index=idx,
        )
        returns[sym] = pd.Series(rng.normal(0, 0.01, n), index=idx)
    feature_df = pd.concat(feature_frames, names=["symbol", "date"])
    return feature_df, returns


def test_target_value_matches_definition():
    horizon = 3
    feature_df, returns = _make_panel(["AAPL"], n=20)
    X, y = build_training_matrix(feature_df, returns, horizon)

    aapl_returns = returns["AAPL"]
    # For row at index t in returns, target should be std of returns.iloc[t+1 : t+4]
    for date in y.loc["AAPL"].index:
        i = aapl_returns.index.get_loc(date)
        expected = float(aapl_returns.iloc[i + 1 : i + 1 + horizon].std(ddof=0))
        actual = float(y.loc[("AAPL", date)])
        assert math.isclose(expected, actual, rel_tol=1e-12), (
            f"date {date}: expected {expected}, got {actual}"
        )
    print(f"target_value: {len(y)} rows verified for AAPL @ horizon={horizon}")


def test_drops_last_h_rows_per_symbol():
    horizon = 5
    n = 30
    feature_df, returns = _make_panel(["AAPL", "MSFT"], n=n)
    X, y = build_training_matrix(feature_df, returns, horizon)

    for sym in ("AAPL", "MSFT"):
        n_rows = len(y.loc[sym])
        assert n_rows == n - horizon, f"{sym}: expected {n - horizon} rows, got {n_rows}"
    print(f"drops_last_h_rows: each symbol has {n - horizon} rows (n={n}, h={horizon})")


def test_pools_multiple_symbols():
    horizon = 3
    feature_df, returns = _make_panel(["AAPL", "MSFT", "SPY"], n=15)
    X, y = build_training_matrix(feature_df, returns, horizon)

    syms = sorted(set(X.index.get_level_values("symbol")))
    assert syms == ["AAPL", "MSFT", "SPY"], f"unexpected symbols: {syms}"
    assert X.index.names == ["symbol", "date"]
    assert len(X) == 3 * (15 - horizon)
    print(f"pools_multiple_symbols: {len(X)} pooled rows across {len(syms)} symbols")


def test_feature_nans_preserved():
    horizon = 3
    feature_df, returns = _make_panel(["AAPL"], n=10)
    feature_df.loc[("AAPL", feature_df.loc["AAPL"].index[5]), "feat_a"] = np.nan
    X, y = build_training_matrix(feature_df, returns, horizon)
    assert X["feat_a"].isna().sum() == 1, "expected exactly 1 NaN in feat_a"
    print("feature_nans: preserved through join (XGBoost handles natively)")


def test_target_nan_rows_dropped():
    """pandas std() is skipna=True, so a target window must be ENTIRELY NaN
    to produce a NaN target. Inject a 3-wide NaN block to force one drop."""
    horizon = 3
    feature_df, returns = _make_panel(["AAPL"], n=10)
    # Indices 4, 5, 6 all NaN → row 3's target window iloc[4:7] is fully NaN
    returns["AAPL"].iloc[4:7] = np.nan
    X, y = build_training_matrix(feature_df, returns, horizon)
    # Pre: 7 rows. Row 3 gets dropped (its target window is fully NaN).
    assert len(y) < 7, f"expected ≥1 row dropped, got {len(y)}"
    row_3_date = feature_df.loc["AAPL"].index[3]
    assert ("AAPL", row_3_date) not in y.index, "row with all-NaN target window should be dropped"
    print(f"target_nan: dropped {7 - len(y)} row(s) with fully-NaN target windows")


def main() -> int:
    test_target_value_matches_definition()
    test_drops_last_h_rows_per_symbol()
    test_pools_multiple_symbols()
    test_feature_nans_preserved()
    test_target_nan_rows_dropped()
    print("all training_matrix tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
