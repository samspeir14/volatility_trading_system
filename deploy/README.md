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

Retrain the h=1 within-stock deviation model every Sunday at 3am UTC: pooled LightGBM (frozen top-10 feature set — earnings/IV/calendar features lead it, selected on eligible-day within-ticker R²) + the HAR-RV benchmark. The job first refreshes daily bars and the DoltHub earnings/IV history CSVs (`data/cache/earnings_history.csv`, `iv_history.csv`) fail-soft, then scores against GARCH/EWMA/persistence baselines on identical walk-forward OOS rows. The job applies the **acceptance gate** (`route` = argmax out-of-sample within-ticker deviation R² over LightGBM / HAR / their 50/50 blend), writes `lgbm_h1_<date>.joblib` + `har_h1_<date>.joblib` + `h1_oos_predictions.parquet` + a schema-v2 `latest_retrain_r2.json` (previous JSON kept as `.bak`), then restarts the bot so the routing takes effect.

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
# Weekly h=1 retrain: tunes the pooled LightGBM, refits the HAR-RV benchmark,
# applies the within-ticker-R² acceptance gate, writes new artifacts +
# latest_retrain_r2.json, then restarts the bot so the routing takes effect.
# The && chain means the restart only fires if the retrain succeeded.
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN_SLOW_TESTS=1
0 3 * * SUN ubuntu cd /home/ubuntu/options-trader && /home/ubuntu/options-trader/venv/bin/python -m tests.test_model_retraining >> /home/ubuntu/options-trader/logs/retrain.log 2>&1 && sudo /bin/systemctl restart options-trader
EOF
```

The `&&` between retraining and restart is load-bearing: if the retrain crashes (bad data, convergence issue, disk full), the bot keeps running on the old artifacts rather than restarting into a broken state.

### 6c. How routing updates flow

1. Cron triggers Sunday 3am UTC.
2. `tests.test_model_retraining` writes `lgbm_h1_<date>.joblib` + `har_h1_<date>.joblib` + `h1_oos_predictions.parquet` + `model/artifacts/latest_retrain_r2.json` (atomic via tempfile + rename; previous JSON preserved as `.bak`).
3. Cron chains `sudo /bin/systemctl restart options-trader` only on retrain exit code 0.
4. Bot startup calls `_load_h1_predictor()`, reads the v2 JSON's `h1.route` (the acceptance-gate verdict), and serves forecasts from LightGBM or HAR accordingly. A missing/legacy/malformed JSON routes to HAR with a loud warning.

A nightly cron (`scripts/reconcile_h1_metrics.py`, e.g. `30 3 * * 2-6`) recomputes every reported h=1 metric from the stored `h1_oos_predictions.parquet` and alerts to Slack if any drifts from the JSON by more than 1e-6 — the routing decision can't silently detach from the predictions it was based on.

The JSON's `h1` block carries per-model **pooled**, **within-ticker**, and **ticker-median deviation R²** plus **level QLIKE**. Within-ticker is the singular metric — the acceptance gate, hyperparameter tuning, and feature-subset selection all optimize it. It strips per-symbol vol levels, so it scores timing skill rather than "TSLA is more volatile than KO". Pooled will read higher; that gap is cross-sectional level credit, not tradeable skill. QLIKE and pooled R² are report-only diagnostics.

## 6d. The strategy and the account profile

The h=1 within-stock deviation model is the only pipeline (VRP harvest retired 2026-08; the legacy multi-horizon 5/10/21 path deleted 2026-08 — `STRATEGY_MODE` and `MODEL_PIPELINE=legacy` both raise a clear startup error if still set; restart to apply).

- **The strategy:** forecast next-day GK vol as a 63-day log-vol baseline + predicted deviation, term-projected along each ticker's GARCH persistence to the option's DTE, compared to that expiration's ATM IV. Entries only at **DTE 1-14** (`MIN_ENTRY_DTE`/`MAX_ENTRY_DTE`) — the shortest-dated options, where a 1-day timing forecast still carries signal; the option-chain scan covers exactly that window. Entry gate ladder (each block is labeled in the signal log and tallied in the daily Gates report): **VRP calibration gate** (per-ticker rolling 252-day history of g = log(IV) − log(tenor-matched realized vol); SELL needs z ≥ +1.5, BUY needs z ≤ −1.25, tickers with <120 gap days emit nothing), **VIX term-structure veto** on every SELL (VIX ≥ VIX3M = crash regime), **model-agreement gate**, **0.25 divergence cap** (hard block on SELL, event-suspect demote on BUY), **earnings entry block** (report anywhere inside the position's life `[today, expiration]`), **macro-event filter** (rate/index-linked names, same life-of-position window), **SPY/QQQ long-straddle exclusion**, liquidity + credit floors, and a **cost gate** (expected edge |forecast − IV| × net vega must cover `COST_MULT`× the sum of half-spreads + fees; any leg quoted wider than `MAX_SPREAD_PCT` of mid blocks outright).
- `ACCOUNT_PROFILE` — `standard` (~$100k calibration, main watchlist) or `small` (~$10k: `watchlist_small.yaml`, `CALIBRATION_SMALL` risk caps, $0.25 min credit, 1-lot sizing).

**Earnings exit (fail-closed):** the exit manager closes every short-vol position at least `EARNINGS_EXIT_BUFFER_DAYS` (default 1) trading days before the ticker's next earnings report — no short-vol position holds through earnings. The next earnings date is stamped on each position at entry and refreshed on every healthy calendar read; if the Finnhub calendar goes stale (>3 days since a successful refresh) the STORED date decides, with a loud journal error. A position with neither a live calendar nor a stored date is flagged for manual review every cycle.

**Deploy blockers before the first h1 sandbox scan:** (1) the retrain JSON must show `h1.acceptance.passed` (or an explicit decision to run `route=har`); (2) `python -m scripts.verify_vega_units` must pass against a live chain, confirming the vega unit convention baked into `signals/cost_gate.py`. Then run `python -m scripts.backfill_vrp_history` once on the box to seed the VRP gap history from the existing divergence log (per-ticker coverage is printed; under-covered tickers stay silent until live logging fills them — visible in the daily Gates block as `vrp_history`).

Every signal (traded or blocked, with its gate label and vrp_z) is logged to `divergence_history.db`, and the daily summary's Gates line (`candidates N | <gate> −k | approved M`) makes a silent zero-signal deadlock visible the same day.

## 6e. Operational guards + dead-man switch

Four guards sit between the strategy and the market (see `risk/trading_guards.py`). All of them block **new entries only** — exit management always keeps running. Every activation is Slack-alerted once and shows up in the `cycle_complete` log line as `entry_blocks=[...]`.

- **Manual HALT**: `touch data/cache/HALT` (optionally write a reason into the file) stops new entries on the next cycle without touching the process. Delete the file to resume. See `deploy/RUNBOOK.md`.
- **Drawdown breakers**: weekly (−8% vs the 5-session equity peak) and monthly (−12% vs the 21-session peak) circuit breakers on top of the daily kill switch, persisted in `risk_state.db` (`breaker_log` table). A tripped breaker blocks entries through the end of its ISO week / calendar month, and re-trips if equity is still deep below the rolling peak after that.
- **Bars freshness**: the guard that turns the frozen-cache failure (2026-04-24→07-02) from "quietly stale" into "loudly blocked". Individually stale symbols are dropped from the entry universe; when ≥50% of the watchlist is stale (bars older than 4 days before the expected end date) the whole entry side blocks.
- **Dead-man switch**: the loop writes `data/cache/heartbeat.json` every iteration (open: per scan cycle; closed: at least hourly). `scripts/heartbeat_check.py` runs from cron *outside* the bot process and Slack-alerts when the heartbeat is >20 min old with the market open (>75 min otherwise). Install:

```bash
sudo tee /etc/cron.d/options-trader-heartbeat >/dev/null <<'EOF'
# Dead-man switch: alert to Slack when the trading loop's heartbeat goes
# stale. Runs every 10 minutes on weekdays around US market hours (UTC).
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
*/10 12-22 * * 1-5 ubuntu cd /home/ubuntu/options-trader && /home/ubuntu/options-trader/venv/bin/python -m scripts.heartbeat_check >> /home/ubuntu/options-trader/logs/heartbeat_check.log 2>&1
EOF
```

The bot also self-reports: 5 consecutive failed cycles post a ⚠️ Slack alert (and a ✅ when cycles recover) — that covers "process alive but can't trade", which the heartbeat alone wouldn't catch.

## 7. Troubleshooting

| Symptom | Check |
|---------|-------|
| Bot exits immediately | `journalctl -u options-trader -n 100` — likely missing artifacts or env vars |
| No trades placed | Risk gates may be rejecting; check `data/cache/risk_state.db` |
| Cycle errors every iteration | Tradier API issue — check rate limit and recent commits |
| Slack posts not arriving | Verify webhook URL with `--summary-only`; check log for 4xx responses |
| Predictions look frozen / stale signals | `SELECT MAX(date) FROM daily_bars` in `data/cache/market_data.db` should be yesterday. The cycle (step 1b) and the retrain both refresh via `ensure_data`; grep the journal for `daily bar refresh failed`. The routing JSON's `bars_through` field records what the last retrain actually trained on |
| Position not closing on exit signal | Confirm `EXECUTE_EXITS` is NOT set to "NO" anywhere; the prod loop always runs live |

## 8. Account safety

The `.env` carries `TRADIER_ENV=sandbox`. To go live (real money), you must change BOTH:
1. `TRADIER_ENV=production`
2. `TRADIER_LIVE_TRADING_CONFIRMED=YES`

Without the second variable, `OrderManager.submit()` refuses to place orders in production mode regardless of the env setting.
