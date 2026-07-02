# Deployment

Production install on the EC2 box. Run as the `ubuntu` user.

## 1. Pre-flight verification

Before installing the systemd unit, verify a single cycle works:

```bash
cd ~/options-trader
source venv/bin/activate
python -m main --once
```

Expected output: a `cycle_complete` log line with current equity, P&L, signal counts. No traceback. Tail `logs/options-trader.log` to confirm the file handler is wired.

## 2. (Optional) Slack webhook

If you want the daily summary posted to Slack, add this to `~/options-trader/.env`:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

Test the format and post path without running a full cycle:

```bash
python -m main --summary-only
```

If no webhook URL is set, the summary is logged to file only — bot keeps running fine.

## 3. Install systemd unit

```bash
sudo cp ~/options-trader/deploy/options-trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable options-trader     # start on boot
sudo systemctl start options-trader
sudo systemctl status options-trader     # confirm it's active
```

## 4. Watch the loop

```bash
journalctl -u options-trader -f          # tail logs in real time
journalctl -u options-trader --since today  # today's logs
tail -f ~/options-trader/logs/options-trader.log  # file output
```

## 5. Stop / restart

```bash
sudo systemctl stop options-trader       # clean stop
sudo systemctl restart options-trader    # restart (after .env or code change)
```

When stopped, open positions persist in the Tradier sandbox. Restart picks up where it left off — exit logic re-evaluates everything on the next cycle.

## 6. Weekly model retraining (cron)

Re-tune LightGBM (primary) and XGBoost (fallback) hyperparameters and refresh saved artifacts every Sunday at 3am UTC. The retraining script saves artifacts for both model families plus a `latest_retrain_r2.json` metadata file that drives `BestPredictor` routing — and then restarts the bot so the new R² values take effect immediately rather than waiting for a manual restart.

### 6a. NOPASSWD sudoers entry for the restart

The cron runs as `ubuntu`, so it needs passwordless sudo for one specific command. Scoped tightly to a single binary + argument:

```bash
sudo tee /etc/sudoers.d/options-trader-restart >/dev/null <<'EOF'
# Allow the ubuntu user to restart the options-trader systemd unit without
# a password. Scoped to that one command so the cron-driven weekly retrain
# can pick up new artifacts without manual intervention.
ubuntu ALL=(root) NOPASSWD: /bin/systemctl restart options-trader
EOF
sudo chmod 0440 /etc/sudoers.d/options-trader-restart
sudo visudo -c -f /etc/sudoers.d/options-trader-restart   # validate before sudo accepts it
```

### 6b. The cron file itself

```bash
sudo tee /etc/cron.d/options-trader-retrain >/dev/null <<'EOF'
# Weekly model retraining: tunes LightGBM (primary) and XGBoost (fallback)
# across all 3 horizons, writes new artifacts + latest_retrain_r2.json, then
# restarts the bot so the new R² values drive routing immediately. The &&
# chain means the restart only fires if the retrain succeeded.
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN_SLOW_TESTS=1
0 3 * * SUN ubuntu cd /home/ubuntu/options-trader && /home/ubuntu/options-trader/venv/bin/python -m tests.test_model_retraining >> /home/ubuntu/options-trader/logs/retrain.log 2>&1 && sudo /bin/systemctl restart options-trader
EOF
```

The `&&` between retraining and restart is load-bearing: if the retrain crashes (bad data, convergence issue, disk full), the bot keeps running on the old artifacts rather than restarting into a broken state.

### 6c. How routing updates flow

1. Cron triggers Sunday 3am UTC.
2. `tests.test_model_retraining` writes `lgbm_h{H}_<date>.joblib` + `xgb_h{H}_<date>.joblib` + `model/artifacts/latest_retrain_r2.json` (atomic via tempfile + rename).
3. Cron chains `sudo /bin/systemctl restart options-trader` only on retrain exit code 0.
4. Bot startup calls `_load_routing_r2()`, reads the JSON, applies the per-horizon R² values to `BestPredictor.update_from_eval`. The `BestPredictor flip h=N: ... -> ...` WARNING in the journal shows when routing changes.

If the JSON is missing, malformed, or partial, `_load_routing_r2` falls back to a conservative hardcoded table — the bot stays bootable while a retrain is pending. Watch for `latest_retrain_r2.json missing` warnings in the journal as a signal that the cron silently failed.

The JSON also carries a `diagnostics_by_horizon` block (informational only — routing ignores it): per-model **within-ticker R²** (per-symbol vol levels stripped out, so it scores timing skill rather than "TSLA is more volatile than KO") and **R² vs a lagged-RV random walk** ("next h days = last h days" — positive means the model beats naive persistence, which is the minimum bar for beating implied vol). Pooled `r2_by_horizon` will read higher than both; that gap is cross-sectional level credit, not tradeable skill.

## 7. Troubleshooting

| Symptom | Check |
|---------|-------|
| Bot exits immediately | `journalctl -u options-trader -n 100` — likely missing artifacts or env vars |
| No trades placed | Risk gates may be rejecting; check `data/cache/risk_state.db` |
| Cycle errors every iteration | Tradier API issue — check rate limit and recent commits |
| Slack posts not arriving | Verify webhook URL with `--summary-only`; check log for 4xx responses |
| Position not closing on exit signal | Confirm `EXECUTE_EXITS` is NOT set to "NO" anywhere; the prod loop always runs live |

## 8. Account safety

The `.env` carries `TRADIER_ENV=sandbox`. To go live (real money), you must change BOTH:
1. `TRADIER_ENV=production`
2. `TRADIER_LIVE_TRADING_CONFIRMED=YES`

Without the second variable, `OrderManager.submit()` refuses to place orders in production mode regardless of the env setting.
