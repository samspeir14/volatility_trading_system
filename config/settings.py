import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

SANDBOX_BASE_URL = "https://sandbox.tradier.com/v1"
PRODUCTION_BASE_URL = "https://api.tradier.com/v1"

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str
    account_id: str
    base_url: str
    env: str
    request_timeout: float = 10.0
    max_retries: int = 3
    # Tradier's hard ceiling is 200 req/min. The per-cycle baseline (option-chain
    # scan + balances + the reconciler's get_positions and per-open-order
    # get_order_status calls added since this budget was first set at 150) grew
    # enough to exhaust the limiter every cycle. 180 restores headroom while
    # staying clear of the 200 ceiling.
    rate_limit_per_min: int = 180
    scan_interval_seconds: int = 300
    expiration_window_days: tuple[int, int] = (14, 45)
    historical_lookback_years: int = 3
    cache_db_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "cache" / "market_data.db"
    )
    finnhub_api_key: str | None = None
    earnings_filter_enabled: bool = True
    earnings_buffer_days: int = 7
    stale_order_threshold_minutes: int = 15
    max_close_retries: int = 3
    # "h1": the within-stock next-day deviation model (the only trading
    # strategy — VRP harvest was retired 2026-08). "legacy": the old
    # multi-horizon (5/10/21) RV models, kept runnable for comparison only.
    model_pipeline: str = "h1"
    # "standard": the ~$100k calibration the watchlist/risk numbers were tuned
    # on. "small": ~$10k bankroll — cheap-ticker watchlist, retuned risk caps.
    # Decoupled from any strategy choice.
    account_profile: str = "standard"
    # VRP calibration gate: SELL needs z >= vrp_z_sell, BUY needs
    # z <= -vrp_z_buy (long-vol loss is bounded, so the bar is lower).
    # A ticker with < vrp_min_obs days of gap history emits no signal.
    vrp_z_sell: float = 1.5
    vrp_z_buy: float = 1.25
    vrp_min_obs: int = 120
    # Cost gate: expected edge (|forecast - IV| x net vega) must cover
    # cost_multiple x total half-spread+fee cost; any leg quoted wider than
    # max_leg_spread_pct of mid blocks entry outright.
    cost_multiple: float = 2.0
    max_leg_spread_pct: float = 0.05
    # Exit rule: close every short-vol position at least this many trading
    # days before the ticker's earnings report.
    earnings_exit_buffer_trading_days: int = 1
    # Estimated all-in cost per option contract per side, netted out of each
    # position's realized P&L. 0.0 preserves gross-P&L bookkeeping (paper).
    # Live on Tradier: ~0.10 covers ORF/OCC/TAF pass-throughs on the Pro plan;
    # ~0.45 on the Lite plan ($0.35 commission + pass-throughs).
    per_contract_fee: float = 0.0


def load_settings() -> Settings:
    load_dotenv()

    env = os.environ.get("TRADIER_ENV", "sandbox").lower()
    if env not in ("sandbox", "production"):
        raise ValueError(f"TRADIER_ENV must be 'sandbox' or 'production', got {env!r}")

    try:
        api_key = os.environ["TRADIER_API_KEY"]
        account_id = os.environ["TRADIER_ACCOUNT_ID"]
    except KeyError as e:
        raise RuntimeError(
            f"Missing required environment variable: {e.args[0]}. "
            "Copy .env.example to .env and fill in your Tradier credentials."
        ) from None

    base_url = os.environ.get("TRADIER_BASE_URL") or (
        SANDBOX_BASE_URL if env == "sandbox" else PRODUCTION_BASE_URL
    )

    finnhub_api_key = os.environ.get("FINNHUB_API_KEY") or None
    earnings_filter_enabled = _parse_bool(
        os.environ.get("EARNINGS_FILTER_ENABLED"), default=True,
    )
    earnings_buffer_days = _parse_int(
        os.environ.get("EARNINGS_BUFFER_DAYS"), default=7,
    )
    stale_order_threshold_minutes = _parse_int(
        os.environ.get("STALE_ORDER_THRESHOLD_MINUTES"), default=15,
    )
    max_close_retries = _parse_int(
        os.environ.get("MAX_CLOSE_RETRIES"), default=3,
    )
    if os.environ.get("STRATEGY_MODE"):
        raise ValueError(
            "STRATEGY_MODE was removed (VRP harvest retired 2026-08). "
            "Use MODEL_PIPELINE=h1|legacy for the model path and "
            "ACCOUNT_PROFILE=standard|small for risk calibration/watchlist."
        )
    model_pipeline = os.environ.get("MODEL_PIPELINE", "h1").strip().lower()
    if model_pipeline not in ("h1", "legacy"):
        raise ValueError(
            f"MODEL_PIPELINE must be 'h1' or 'legacy', got {model_pipeline!r}"
        )
    account_profile = os.environ.get("ACCOUNT_PROFILE", "standard").strip().lower()
    if account_profile not in ("standard", "small"):
        raise ValueError(
            f"ACCOUNT_PROFILE must be 'standard' or 'small', got {account_profile!r}"
        )
    vrp_z_sell = _parse_float(os.environ.get("VRP_Z_SELL"), default=1.5)
    vrp_z_buy = _parse_float(os.environ.get("VRP_Z_BUY"), default=1.25)
    # Both knobs are magnitudes — the BUY gate applies the minus sign itself
    # (z <= -vrp_z_buy). Someone writing VRP_Z_BUY=-1.25 (the natural reading
    # of "BUY needs z <= -1.25") would silently turn the gate into z <= +1.25
    # and make almost everything a BUY; reject non-positive values instead.
    if vrp_z_sell <= 0 or vrp_z_buy <= 0:
        raise ValueError(
            f"VRP_Z_SELL/VRP_Z_BUY are magnitudes and must be > 0 "
            f"(the BUY gate is z <= -VRP_Z_BUY); got "
            f"{vrp_z_sell}/{vrp_z_buy}"
        )
    vrp_min_obs = _parse_int(os.environ.get("VRP_MIN_OBS"), default=120)
    cost_multiple = _parse_float(os.environ.get("COST_MULT"), default=2.0)
    max_leg_spread_pct = _parse_float(os.environ.get("MAX_SPREAD_PCT"), default=0.05)
    if cost_multiple <= 0 or max_leg_spread_pct <= 0:
        raise ValueError(
            f"COST_MULT and MAX_SPREAD_PCT must be > 0, got "
            f"{cost_multiple}/{max_leg_spread_pct}"
        )
    earnings_exit_buffer_trading_days = _parse_int(
        os.environ.get("EARNINGS_EXIT_BUFFER_DAYS"), default=1,
    )
    # Strict parse, unlike the operational knobs above: a typo'd fee silently
    # becoming 0.0 would overstate every live realized P&L with no log trace.
    raw_fee = os.environ.get("PER_CONTRACT_FEE")
    if raw_fee is None or raw_fee.strip() == "":
        per_contract_fee = 0.0
    else:
        try:
            per_contract_fee = float(raw_fee)
        except ValueError:
            raise ValueError(
                f"PER_CONTRACT_FEE must be a number, got {raw_fee!r}"
            ) from None
    if per_contract_fee < 0:
        raise ValueError(f"PER_CONTRACT_FEE must be >= 0, got {per_contract_fee}")

    return Settings(
        api_key=api_key,
        account_id=account_id,
        base_url=base_url.rstrip("/"),
        env=env,
        finnhub_api_key=finnhub_api_key,
        earnings_filter_enabled=earnings_filter_enabled,
        earnings_buffer_days=earnings_buffer_days,
        stale_order_threshold_minutes=stale_order_threshold_minutes,
        max_close_retries=max_close_retries,
        model_pipeline=model_pipeline,
        account_profile=account_profile,
        vrp_z_sell=vrp_z_sell,
        vrp_z_buy=vrp_z_buy,
        vrp_min_obs=vrp_min_obs,
        cost_multiple=cost_multiple,
        max_leg_spread_pct=max_leg_spread_pct,
        earnings_exit_buffer_trading_days=earnings_exit_buffer_trading_days,
        per_contract_fee=per_contract_fee,
    )


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _parse_int(raw: str | None, *, default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float(raw: str | None, *, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default
