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

## 6d. Strategy mode

`STRATEGY_MODE` in `.env` selects what the bot trades (restart to apply):

- `model` (default): the original strategy — trade the model-vs-IV divergence in both directions, gated by z-score, divergence cap, and earnings filter.
- `harvest`: sell short-DTE iron condors (entry DTE 5–15, nearest-first so the book concentrates at DTE 5–8 and rides toward expiry) on every eligible watchlist name, every cycle — the variance-risk-premium harvesting strategy motivated by the 2026-07 research: the short-tenor premium is fat and unconditional, and nothing (model, formula, or IV-gap rule) ordered it out-of-sample. Entry gates that remain: earnings filter, liquidity filters, an **extreme-spread veto** (skip when ATM IV exceeds trailing 63-day realized vol by more than 0.12 — big gaps historically meant the market was pricing real incoming vol, not extra premium; March 2020 shape), a **VIX term-structure veto** (VIX ≥ VIX3M = backwardation = crash regime; all SELL entries demoted until the curve normalizes), a **macro-event filter** (FOMC/CPI dates inside the position's life demote the rate/index-linked names — TLT, SLV, index ETFs — for whom those releases are effectively earnings; single-name equities are not macro-gated since the 7-year VRP was fat through every macro day; hand-maintained table in `data/macro_calendar.py`, extend yearly), and a **credit-to-width floor** (mid credit must be ≥25% of wing width). No BUY side. The thesis-reversal exit is disabled (there is no model thesis); stop-loss, profit-target, and assignment-risk exits stay live.
- `small_harvest`: the harvest strategy recalibrated for a **~$10k bankroll** — see the paragraph at the end of this section.

Sizing is built for the correlated crash, not the average week: **1.5% equity max loss per trade** and a **20% portfolio wing-risk cap** (sum of open max-losses — the book's worst case when every name gaps through its wings at once — bounded by construction; entries auto-throttle when the ladder is full). The 20% is **paper calibration** — faster friction measurement, fake drawdowns are tuition; revisit down to ≤12% before any real-money deployment. Risk gates also accrue approvals *within* a cycle, so a first-day batch of harvest signals can't stampede past the portfolio/sector/ticker caps. Note at 1.5%/$100k the expensive names (AMD, CAT, GS: price × IV too big for one contract) never trade — they already didn't fit the old 2% budget — and META/TSLA/UNH only enter at the short end of the DTE window.

In both modes the models run every cycle and every signal (traded or not) is logged to `divergence_history.db`, so model-accuracy evaluations (e.g. the model-vs-IV-vs-trail63 prospective test) continue regardless of what's being traded.

Harvest mode also narrows the scan's expiration window to (3, 16) days — entries cap at DTE 15 and positions ride to expiry, so longer chains are dead weight, and the saved API calls pay for the larger watchlist (33 names as of 2026-07) against the 180/min rate budget. Open positions above the window (e.g. legacy model-mode condors during a transition) still mark via `fetch_missing_position_chains`. Note the divergence log only accumulates DTE ≤ 16 rows while in harvest mode. Watchlist candidates are screened with `experiments/screen_watchlist_candidates.py` (run intraday — off-hours quotes fake out the spread/OI filters; note its `BUDGET` constant is $1,500 = 1.5% × $100k — rerun with the profile's actual budget when screening for small_harvest).

`small_harvest` is the harvest strategy recalibrated for a **~$10k bankroll** — the intended first live-money configuration. Same code path as `harvest` (internally it normalizes to `strategy_mode="harvest"` with `harvest_profile="small"`); what changes is the calibration bundle (`CALIBRATION_SMALL` in `main.py`) and the universe (`config/watchlist_small.yaml`, 19 cheap liquid weekly-options names across 11 sectors, verified 2026-07). Orders are 1-lot, so the per-trade budget is an *eligibility gate*: at **2.5%/$10k = $250** the whole small watchlist's 1σ condors fit ($40–$220 max loss), with the priciest (BAC/SLV/B) fitting at the short-DTE end only. Portfolio wing-risk cap is **12%** — the real-money number the harvest paragraph above defers to. The gamma cap is **10%**, not 1%: the gate sums raw gamma × 100 and raw ATM gamma scales as 1/(S·σ·√T), so cheap names carry ~15× the raw gamma of $100k-watchlist names — at 1% the cap would choke the book after ~2 positions. New floors for small-credit economics: signals need **$0.25 mid credit** minimum, and the premium backstop drops to $500. Set `PER_CONTRACT_FEE` in `.env` when live (0.10 on Tradier Pro pass-throughs; 0.45 on Lite) — it's netted out of realized P&L (close = both sides' legs, expiry = open side only), keeping the realized/unrealized split honest since Tradier's equity already reflects fees. Before going live: Tradier margin accounts need **$2,000 minimum** and options **Level 3** (spreads); confirm with Tradier that day-trade counting is off post-FINRA-26-10 (PDT was eliminated 2026-06-04, but brokers may phase in until 2027-10) since profit-target exits can close same-day; and subscribe to Tradier Pro ($10/mo) — at ~$0.35/contract on Lite, 8-contract round trips eat 10–15% of small-condor winners.

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
