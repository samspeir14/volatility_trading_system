"""Dead-man switch for the trading loop, run from cron OUTSIDE the bot process.

The bot writes data/cache/heartbeat.json on every run_forever iteration
(market open: every scan cycle, ~5 min; market closed: at least hourly). This
script alerts to Slack when the heartbeat goes stale — the classic silent
failure is the EC2 process dying with positions still open.

Alert thresholds:
  * last heartbeat said market_state="open"  → stale after 20 minutes
  * anything else (closed / error / unknown) → stale after 75 minutes
    (the closed-market loop re-polls at most hourly, so >75 min means the
    process is gone, not sleeping)

Deduping: at most one alert per stuck heartbeat per REALERT_HOURS, tracked in
heartbeat_alert_state.json next to the heartbeat.

Cron (EC2 box, UTC; every 10 min on weekdays around US market hours):
  */10 12-22 * * 1-5  cd /home/ubuntu/options_trader && \
      .venv/bin/python -m scripts.heartbeat_check >> logs/heartbeat_check.log 2>&1
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT_PATH = PROJECT_ROOT / "data" / "cache" / "heartbeat.json"
ALERT_STATE_PATH = PROJECT_ROOT / "data" / "cache" / "heartbeat_alert_state.json"

STALE_OPEN_SECONDS = 20 * 60
STALE_ANY_SECONDS = 75 * 60
REALERT_HOURS = 6.0


def should_alert(heartbeat: dict | None, now: datetime) -> str | None:
    """Pure decision logic (unit-tested): reason string if the heartbeat is
    missing/stale, None if healthy."""
    if heartbeat is None:
        return (
            f"heartbeat file missing at {HEARTBEAT_PATH} — bot never started "
            "or crashed before its first cycle"
        )
    try:
        ts = datetime.fromisoformat(str(heartbeat.get("ts")))
    except (TypeError, ValueError):
        return f"heartbeat unparsable: {heartbeat!r}"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (now - ts).total_seconds()
    state = heartbeat.get("market_state", "unknown")
    limit = STALE_OPEN_SECONDS if state == "open" else STALE_ANY_SECONDS
    if age > limit:
        return (
            f"trading loop heartbeat is stale: last beat {age / 60:.0f} min ago "
            f"(market_state={state}, limit {limit / 60:.0f} min). "
            "Process is likely down — see deploy/RUNBOOK.md"
        )
    return None


def _already_alerted(heartbeat_ts: str | None, now: datetime) -> bool:
    """True when we alerted for this same stuck heartbeat within REALERT_HOURS."""
    try:
        state = json.loads(ALERT_STATE_PATH.read_text())
        last_alert = datetime.fromisoformat(state["alerted_at"])
        same_beat = state.get("heartbeat_ts") == heartbeat_ts
        recent = (now - last_alert).total_seconds() < REALERT_HOURS * 3600
        return same_beat and recent
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _record_alert(heartbeat_ts: str | None, now: datetime) -> None:
    try:
        ALERT_STATE_PATH.write_text(json.dumps({
            "heartbeat_ts": heartbeat_ts,
            "alerted_at": now.isoformat(),
        }))
    except OSError:
        pass


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    from risk.trading_guards import read_heartbeat

    now = datetime.now(timezone.utc)
    heartbeat = read_heartbeat(HEARTBEAT_PATH)
    reason = should_alert(heartbeat, now)
    if reason is None:
        print(f"{now.isoformat()} heartbeat healthy")
        return 0

    heartbeat_ts = heartbeat.get("ts") if heartbeat else None
    if _already_alerted(heartbeat_ts, now):
        print(f"{now.isoformat()} still stale, already alerted: {reason}")
        return 1

    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        from logs.slack import post_text

        post_text(webhook, f"💀 DEAD-MAN SWITCH: {reason}")
    else:
        print("SLACK_WEBHOOK_URL not set — cannot alert", file=sys.stderr)
    print(f"{now.isoformat()} ALERT: {reason}", file=sys.stderr)
    _record_alert(heartbeat_ts, now)
    return 1


if __name__ == "__main__":
    sys.exit(main())
