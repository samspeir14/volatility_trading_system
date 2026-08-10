import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from signals import DivergenceHistory


@dataclass
class FakeSignal:
    symbol: str
    expiration: date
    horizon_lower: int
    horizon_upper: int
    weight_lower: float
    predicted_iv_equivalent: float
    atm_iv: float
    divergence: float
    underlying_price: float
    is_actionable: bool = True
    vrp_z: float | None = None
    blocked_by: str | None = None


def _fake(symbol: str, exp_offset: int, divergence: float) -> FakeSignal:
    return FakeSignal(
        symbol=symbol,
        expiration=date(2026, 5, 1) + timedelta(days=exp_offset),
        horizon_lower=10,
        horizon_upper=21,
        weight_lower=0.5,
        predicted_iv_equivalent=0.30,
        atm_iv=0.30 - divergence,
        divergence=divergence,
        underlying_price=200.0,
    )


def test_empty_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = DivergenceHistory(Path(tmp) / "h.db")
        z = store.time_series_z_score("AAPL", 10, 21, 0.05)
        assert z is None
        store.close()
    print("empty: time_series_z_score returns None")


def test_log_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        store = DivergenceHistory(Path(tmp) / "h.db")
        signals = [_fake("AAPL", i, 0.01 * i) for i in range(10)]
        n = store.log_signals(signals, scan_date=date(2026, 4, 1))
        assert n == 10
        assert store.row_count() == 10
        store.close()
    print("log_and_count: 10 rows logged and counted")


def test_z_score_against_history():
    with tempfile.TemporaryDirectory() as tmp:
        store = DivergenceHistory(Path(tmp) / "h.db")
        # Insert 30 historical divergences across 30 dates (one per day, all same symbol/horizons)
        for i in range(30):
            sig = _fake("AAPL", 0, 0.01 * (i + 1))  # divergences from 0.01 to 0.30
            store.log_signals([sig], scan_date=date(2026, 4, 1) + timedelta(days=i))
        # Manually compute expected z for current_divergence = 0.20
        rows = [0.01 * (i + 1) for i in range(30)]
        mean = sum(rows) / len(rows)
        var = sum((x - mean) ** 2 for x in rows) / len(rows)
        std = math.sqrt(var)
        current = 0.20
        expected_z = (current - mean) / std

        actual_z = store.time_series_z_score("AAPL", 10, 21, current)
        assert actual_z is not None
        assert math.isclose(actual_z, expected_z, rel_tol=1e-9), (
            f"expected {expected_z}, got {actual_z}"
        )
        store.close()
    print(f"z_score: matches manual calc to 1e-9 (n=30, z={actual_z:.4f})")


def test_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "h.db"
        store1 = DivergenceHistory(path)
        for i in range(25):
            sig = _fake("AAPL", 0, 0.01 * i)
            store1.log_signals([sig], scan_date=date(2026, 4, 1) + timedelta(days=i))
        store1.close()

        store2 = DivergenceHistory(path)
        assert store2.row_count() == 25
        z = store2.time_series_z_score("AAPL", 10, 21, 0.10)
        assert z is not None
        store2.close()
    print("persistence: 25 rows survive close/reopen, z still computable")


def test_lookback_truncation():
    with tempfile.TemporaryDirectory() as tmp:
        store = DivergenceHistory(Path(tmp) / "h.db")
        # 100 rows, divergences increasing by date
        for i in range(100):
            sig = _fake("AAPL", 0, 0.001 * i)
            store.log_signals([sig], scan_date=date(2026, 1, 1) + timedelta(days=i))

        # With lookback=20, we should use only the most recent 20 (days 80-99 → divergences 0.080-0.099)
        z_20 = store.time_series_z_score("AAPL", 10, 21, 0.05, lookback=20)
        # With lookback=100, we use all 100 rows (mean closer to 0.05)
        z_100 = store.time_series_z_score("AAPL", 10, 21, 0.05, lookback=100)
        # The two z-scores should differ — different reference distributions
        assert z_20 is not None and z_100 is not None
        assert not math.isclose(z_20, z_100, abs_tol=0.01), (
            f"lookback should change z: z20={z_20}, z100={z_100}"
        )
        store.close()
    print(f"lookback: z@20={z_20:.3f} differs from z@100={z_100:.3f}")


def main() -> int:
    test_empty_returns_none()
    test_log_and_count()
    test_z_score_against_history()
    test_persistence_across_reopen()
    test_lookback_truncation()
    print("all divergence_history tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
