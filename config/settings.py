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
    # "model": trade model-vs-IV divergence (both directions, z-gated).
    # "harvest": sell short-DTE iron condors on every eligible name — the
    # variance-risk-premium harvesting strategy; model runs log-only.
    strategy_mode: str = "model"


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
    strategy_mode = os.environ.get("STRATEGY_MODE", "model").strip().lower()
    if strategy_mode not in ("model", "harvest"):
        raise ValueError(
            f"STRATEGY_MODE must be 'model' or 'harvest', got {strategy_mode!r}"
        )

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
        strategy_mode=strategy_mode,
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
