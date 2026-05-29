"""Unit tests for MarketData.fetch_missing_position_chains — pulling chains for
open-position expirations that aged below the scan window so mark-to-market and
the expiration-proximity exit don't silently drop them."""
import asyncio
from datetime import date, datetime, timezone
from unittest import mock

from config import Ticker
from data.async_client import OptionContract
from data.market_data import MarketData, ScanResult, TickerSnapshot


def _contract(symbol: str, underlying: str, expiration: date, strike: float,
              option_type: str) -> OptionContract:
    return OptionContract(
        symbol=symbol, underlying=underlying, expiration=expiration,
        strike=strike, option_type=option_type, bid=1.0, ask=1.2, last=1.1,
        volume=10, open_interest=100, delta=0.5, gamma=0.1, theta=-0.05,
        vega=0.2, iv=0.30, fetched_at=datetime.now(timezone.utc),
    )


def _scan(snapshots: dict[str, TickerSnapshot]) -> ScanResult:
    return ScanResult(
        fetched_at=datetime(2026, 5, 29, 15, 0, tzinfo=timezone.utc),
        snapshots=snapshots,
    )


def _market_data(get_chain_side_effect):
    client = mock.AsyncMock()
    client.get_chain.side_effect = get_chain_side_effect
    watchlist = [Ticker(symbol="AAPL", sector="tech"), Ticker(symbol="SPY", sector="etf")]
    return MarketData(client, watchlist), client


def test_below_window_expiration_is_fetched_and_merged():
    today = date(2026, 5, 29)
    near_exp = date(2026, 5, 30)  # DTE 1 — below a (3, 45) window
    near_leg = _contract("AAPL260530C00150000", "AAPL", near_exp, 150.0, "call")

    async def fake_chain(symbol, expiration, greeks=True):
        assert (symbol, expiration) == ("AAPL", near_exp)
        return [near_leg]

    md, client = _market_data(fake_chain)
    # Scan only has a far-dated AAPL contract, not the near-expiry leg.
    far_leg = _contract("AAPL260619C00150000", "AAPL", date(2026, 6, 19), 150.0, "call")
    scan = _scan({"AAPL": TickerSnapshot("AAPL", "tech", {"last": 150.0}, [far_leg])})

    result = asyncio.run(md.fetch_missing_position_chains(
        scan, {"AAPL": {near_exp}}, today=today,
    ))

    symbols = {c.symbol for c in result["AAPL"].contracts}
    assert "AAPL260530C00150000" in symbols, "near-expiry leg should be merged in"
    assert "AAPL260619C00150000" in symbols, "existing scan contracts preserved"
    # Underlying quote from the original snapshot is preserved.
    assert result["AAPL"].underlying == {"last": 150.0}
    client.get_chain.assert_awaited_once()
    print("below-window expiration fetched and merged")


def test_already_covered_expiration_not_refetched():
    today = date(2026, 5, 29)
    covered_exp = date(2026, 6, 19)
    leg = _contract("AAPL260619C00150000", "AAPL", covered_exp, 150.0, "call")

    async def fake_chain(symbol, expiration, greeks=True):
        raise AssertionError("should not fetch an expiration already in the scan")

    md, client = _market_data(fake_chain)
    scan = _scan({"AAPL": TickerSnapshot("AAPL", "tech", {}, [leg])})

    result = asyncio.run(md.fetch_missing_position_chains(
        scan, {"AAPL": {covered_exp}}, today=today,
    ))

    assert result is scan, "no fetch needed → same scan object returned"
    client.get_chain.assert_not_awaited()
    print("already-covered expiration not re-fetched")


def test_expired_expiration_not_fetched():
    today = date(2026, 5, 29)
    expired_exp = date(2026, 5, 22)  # in the past — reconciler's job

    async def fake_chain(symbol, expiration, greeks=True):
        raise AssertionError("should not fetch an already-expired expiration")

    md, client = _market_data(fake_chain)
    scan = _scan({"AAPL": TickerSnapshot("AAPL", "tech", {}, [])})

    result = asyncio.run(md.fetch_missing_position_chains(
        scan, {"AAPL": {expired_exp}}, today=today,
    ))

    assert result is scan
    client.get_chain.assert_not_awaited()
    print("expired expiration left to the reconciler, not fetched")


def test_empty_needed_is_noop():
    md, client = _market_data(lambda *a, **k: [])
    scan = _scan({"AAPL": TickerSnapshot("AAPL", "tech", {}, [])})
    result = asyncio.run(md.fetch_missing_position_chains(scan, {}, today=date(2026, 5, 29)))
    assert result is scan
    client.get_chain.assert_not_awaited()
    print("empty needed → no-op")


def test_symbol_absent_from_scan_gets_new_snapshot():
    """Defensive: if an open position's symbol somehow isn't in the scan, a
    fresh snapshot is created (sector resolved from the watchlist map)."""
    today = date(2026, 5, 29)
    near_exp = date(2026, 5, 30)
    leg = _contract("SPY260530P00400000", "SPY", near_exp, 400.0, "put")

    async def fake_chain(symbol, expiration, greeks=True):
        return [leg]

    md, client = _market_data(fake_chain)
    scan = _scan({"AAPL": TickerSnapshot("AAPL", "tech", {}, [])})

    result = asyncio.run(md.fetch_missing_position_chains(
        scan, {"SPY": {near_exp}}, today=today,
    ))

    assert "SPY" in result.snapshots
    assert result["SPY"].sector == "etf"
    assert [c.symbol for c in result["SPY"].contracts] == ["SPY260530P00400000"]
    print("missing symbol gets a fresh snapshot with watchlist sector")


def _run_all():
    test_below_window_expiration_is_fetched_and_merged()
    test_already_covered_expiration_not_refetched()
    test_expired_expiration_not_fetched()
    test_empty_needed_is_noop()
    test_symbol_absent_from_scan_gets_new_snapshot()
    print("all scan-augmentation tests passed")


if __name__ == "__main__":
    _run_all()
