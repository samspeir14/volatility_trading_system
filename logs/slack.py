from __future__ import annotations

import logging

import requests

from logs.daily_summary import DailySummary

logger = logging.getLogger(__name__)


def format_summary(summary: DailySummary) -> str:
    eq_change = summary.equity_change
    eq_pct = (eq_change / summary.starting_equity * 100) if summary.starting_equity > 0 else 0.0

    lines = [
        f"*Options Trader — {summary.date.isoformat()}*",
        f"Equity: ${summary.starting_equity:,.2f} → ${summary.ending_equity:,.2f} "
        f"({eq_change:+,.2f}, {eq_pct:+.2f}%)",
        f"P&L today: realized ${summary.realized_pnl:+,.2f} / "
        f"unrealized ${summary.unrealized_pnl:+,.2f} "
        f"(total ${summary.total_pnl:+,.2f}, {summary.total_pnl_pct:+.2%})",
        f"Positions: {summary.open_positions} open • "
        f"{summary.positions_opened_today} opened • "
        f"{summary.positions_closed_today} closed",
        f"Signals: {summary.signals_total} raw → "
        f"{summary.signals_approved} approved",
    ]

    if summary.risk_rejections_total > 0:
        reason_str = ", ".join(
            f"{cat} × {n}"
            for cat, n in sorted(
                summary.risk_rejections_by_reason.items(),
                key=lambda kv: -kv[1],
            )
        )
        lines.append(f"Risk rejections: {summary.risk_rejections_total} ({reason_str})")
    else:
        lines.append("Risk rejections: 0")

    if summary.top_exit_triggers:
        trig_str = ", ".join(
            f"{trig} × {n}"
            for trig, n in sorted(
                summary.top_exit_triggers.items(),
                key=lambda kv: -kv[1],
            )
        )
        lines.append(f"Exits: {trig_str}")

    if summary.kill_switch_activated:
        lines.append(":rotating_light: *Kill switch ACTIVATED today*")
    else:
        lines.append("Kill switch: not activated")

    if summary.earnings_straddling_positions:
        lines.append(
            f":warning: {len(summary.earnings_straddling_positions)} open "
            "position(s) straddle an earnings event:"
        )
        for p in summary.earnings_straddling_positions:
            lines.append(
                f"  • {p.symbol} {p.structure} exp {p.expiration.isoformat()} "
                f"(earnings {p.earnings_date.isoformat()})"
            )

    return "\n".join(lines)


def post_to_slack(webhook_url: str, summary: DailySummary, timeout: float = 10.0) -> bool:
    """Returns True on successful post, False otherwise. Never raises."""
    text = format_summary(summary)
    try:
        resp = requests.post(webhook_url, json={"text": text}, timeout=timeout)
        if resp.ok:
            logger.info("Slack daily summary posted (%d chars)", len(text))
            return True
        logger.warning("Slack post returned %d: %s", resp.status_code, resp.text[:200])
        return False
    except requests.RequestException as e:
        logger.warning("Slack post failed: %s", e)
        return False
