"""Nightly reconciliation of the h=1 retrain metrics, run from cron OUTSIDE
the bot process.

Recomputes the per-model pooled deviation R² (and the other reported h=1
metrics) from the stored OOS predictions (model/artifacts/
h1_oos_predictions.parquet) using the exact same code path as the retrain
job (model.evaluation.h1_metrics_from_predictions), and asserts each value
matches latest_retrain_r2.json within 1e-6. A mismatch means the stored
predictions and the reported numbers have drifted apart — a corrupted
artifact, a partial write, or a code change that silently altered the
metric — and the routing decision can no longer be trusted.

Exit codes: 0 ok / skipped (legacy pipeline), 1 mismatch or missing input.
Posts to Slack via SLACK_WEBHOOK_URL on breach.

Cron (EC2 box, UTC; nightly on weekdays):
  30 3 * * 2-6  cd /home/ubuntu/options_trader && \
      .venv/bin/python -m scripts.reconcile_h1_metrics >> logs/reconcile_h1.log 2>&1
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts"
ROUTING_R2_JSON = ARTIFACT_DIR / "latest_retrain_r2.json"

TOLERANCE = 1e-6
CHECKED_METRICS = (
    "deviation_r2_pooled",
    "deviation_r2_within",
    "deviation_r2_ticker_median",
    "qlike_level",
)
# JSON metric key -> key in h1_metrics_from_predictions output
_METRIC_MAP = {
    "deviation_r2_pooled": "dev_r2_pooled",
    "deviation_r2_within": "dev_r2_within",
    "deviation_r2_ticker_median": "dev_r2_ticker_median",
    "qlike_level": "qlike_level",
}


def reconcile(payload: dict, preds_long) -> list[str]:
    """Pure comparison logic (unit-tested): list of mismatch descriptions,
    empty when everything reconciles."""
    from model.evaluation import h1_metrics_from_predictions

    recomputed = h1_metrics_from_predictions(preds_long)
    problems: list[str] = []
    h1 = payload.get("h1", {})
    for metric_key in CHECKED_METRICS:
        reported_by_model = h1.get(metric_key, {})
        for model_name, reported in reported_by_model.items():
            if model_name not in recomputed:
                problems.append(
                    f"{metric_key}[{model_name}]: reported but model absent "
                    f"from stored predictions"
                )
                continue
            actual = recomputed[model_name][_METRIC_MAP[metric_key]]
            if abs(float(reported) - float(actual)) > TOLERANCE:
                problems.append(
                    f"{metric_key}[{model_name}]: reported {reported!r} vs "
                    f"recomputed {actual!r} (Δ > {TOLERANCE})"
                )
    return problems


def _alert(text: str) -> None:
    print(text, file=sys.stderr)
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        from logs.slack import post_text

        post_text(webhook, f":rotating_light: h1 metric reconciliation: {text}")
    else:
        print("SLACK_WEBHOOK_URL not set — cannot alert", file=sys.stderr)


def main() -> int:
    import pandas as pd

    if not ROUTING_R2_JSON.exists():
        _alert(f"routing JSON missing at {ROUTING_R2_JSON}")
        return 1
    payload = json.loads(ROUTING_R2_JSON.read_text())
    if payload.get("pipeline") != "h1":
        print(f"pipeline={payload.get('pipeline')!r} — nothing to reconcile")
        return 0

    preds_file = payload.get("h1", {}).get("predictions_file")
    if not preds_file:
        _alert("h1 payload has no predictions_file")
        return 1
    preds_path = ARTIFACT_DIR / preds_file
    if not preds_path.exists():
        _alert(f"stored predictions missing at {preds_path}")
        return 1

    preds_long = pd.read_parquet(preds_path)
    problems = reconcile(payload, preds_long)
    if problems:
        _alert(
            f"{len(problems)} mismatch(es) between stored predictions and "
            f"reported metrics: " + "; ".join(problems[:5])
        )
        return 1
    n_models = len(payload.get("h1", {}).get("qlike_level", {}))
    print(
        f"reconciled {len(CHECKED_METRICS)} metrics x "
        f"{n_models} models within {TOLERANCE} — OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
