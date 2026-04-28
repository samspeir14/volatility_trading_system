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

Re-tune XGBoost hyperparameters and refresh saved artifacts every Sunday at 3am UTC. The slow tuning test already saves artifacts when invoked.

```bash
sudo tee /etc/cron.d/options-trader-retrain <<'EOF'
0 3 * * SUN ubuntu cd /home/ubuntu/options-trader && /home/ubuntu/options-trader/venv/bin/python -m tests.test_xgb_hyperparam_tuning >> /home/ubuntu/options-trader/logs/retrain.log 2>&1
EOF
```

Set `RUN_SLOW_TESTS=1` in `.env` so the test isn't skipped. The bot's bootstrap loader picks up the newest artifact by mtime on the next Monday startup.

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
