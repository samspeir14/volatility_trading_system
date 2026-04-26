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
    rate_limit_per_min: int = 150
    scan_interval_seconds: int = 300
    expiration_window_days: tuple[int, int] = (14, 45)
    historical_lookback_years: int = 3
    cache_db_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "cache" / "market_data.db"
    )


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

    return Settings(
        api_key=api_key,
        account_id=account_id,
        base_url=base_url.rstrip("/"),
        env=env,
    )
