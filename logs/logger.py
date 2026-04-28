from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(
    log_dir: Path | str = Path("logs"),
    level: str = "INFO",
    file_name: str = "options-trader.log",
) -> None:
    """Configure root logger with rotating file + console output. Idempotent."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        # Already configured — don't double-attach
        return

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = TimedRotatingFileHandler(
        log_dir / file_name,
        when="midnight",
        backupCount=14,
        utc=True,
    )
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    root.setLevel(level.upper())
    root.addHandler(file_handler)
    root.addHandler(console)
