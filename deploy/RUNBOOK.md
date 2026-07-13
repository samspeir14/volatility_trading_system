# Runbook — when the bot misbehaves

Procedures for operating the bot under pressure. Written to be followed
verbatim at 3pm on a bad day; read it once now so you know what's here.

All paths are on the EC2 box (`~/options-trader`), run as `ubuntu`.

## 0. Decide: pause or flatten?

- **Pause** (stop NEW entries, keep managing exits): use the HALT flag. This
  is almost always the right first move — the exit logic keeps working
  stops/targets/assignment closes while you investigate.
- **Flatten** (get out of everything): section 2. Only for "the bot is doing
  something actively wrong to positions" or a broker/margin emergency.

## 1. Pause new entries (HALT flag)

```bash
echo "why you halted, your name, date" > ~/options-trader/data/cache/HALT
```

Takes effect on the next cycle (≤5 min): Slack posts one "🛑 Entries blocked"
message, `cycle_complete` lines show `entry_blocks=[manual HALT: ...]`, exits
keep running. **Resume:**

```bash
rm ~/options-trader/data/cache/HALT
```

## 2. Flatten everything (manual)

1. **Stop the bot first** — otherwise it may re-enter or fight your closes:
   ```bash
   sudo systemctl stop options-trader
   ```
2. **Cancel working orders** in the Tradier dashboard (dash.tradier.com →
   Accounts → Orders): cancel anything in `open`/`pending` state. Entry orders
   first (so nothing new fills), then stale closes.
3. **Close positions** in the dashboard, per position: multileg close of all
   legs at once (don't leg out of condors one side at a time — closing the
   shorts first is fine, closing the LONGS first leaves you naked short).
   Work the price: start at mid, give it 30–60s, step one cent toward the
   market. Priority order if time-constrained:
   1. anything expiring today or tomorrow,
   2. positions with a short leg in the money,
   3. everything else.
4. **Reconcile after**: when the bot restarts, `PositionReconciler` marks
   log rows closed against Tradier's positions endpoint on the first cycle.
   Check the first `cycle_complete` line and Slack summary that evening.

## 3. Clear a tripped guard

### Daily kill switch (auto-resets next trading day)

Rarely needs manual clearing. If you must resume the same day:

```bash
sqlite3 ~/options-trader/data/cache/risk_state.db \
  "DELETE FROM kill_switch_log WHERE date = date('now');"
```

Think twice: it tripped because the day lost ≥5%.

### Weekly / monthly drawdown breaker

Auto-expires at the end of the ISO week / calendar month, but re-trips if
equity is still ≥8%/12% below the rolling peak. Inspect and clear early:

```bash
sqlite3 ~/options-trader/data/cache/risk_state.db \
  "SELECT kind, date, reason FROM breaker_log ORDER BY date DESC LIMIT 5;"
sqlite3 ~/options-trader/data/cache/risk_state.db \
  "DELETE FROM breaker_log WHERE kind='weekly' AND date='YYYY-MM-DD';"
```

### Bars-freshness block

Don't clear it — fix the data. It means the daily-bars cache is stale:

```bash
sqlite3 ~/options-trader/data/cache/market_data.db \
  "SELECT symbol, MAX(date) FROM daily_bars GROUP BY symbol ORDER BY 2 LIMIT 10;"
journalctl -u options-trader --since today | grep "daily bar refresh"
```

The refresh self-heals on the next successful cycle (ensure_data is
incremental); the block lifts by itself once bars are current.

## 4. Dead-man alert fired ("💀 DEAD-MAN SWITCH")

The heartbeat went stale — the process is down or wedged.

```bash
sudo systemctl status options-trader        # dead? crashed? restarting loop?
journalctl -u options-trader -n 200         # last words
df -h / && free -m                          # disk full / OOM are the classics
sudo systemctl restart options-trader
journalctl -u options-trader -f             # watch first cycle complete
```

If it won't stay up and the market is open: HALT flag (harmless while down),
then treat open positions per section 2 if any are near expiry — the
assignment-risk exit can't protect positions while the bot is down.

Known noise after any restart: a burst of "Rate limiter exhausted" warnings
on the first cycles is benign (account-wide budget refilling), not a bug.

## 5. "⚠️ N consecutive cycle errors" alert

Bot is up but cycles keep failing (usually Tradier API or network).

```bash
journalctl -u options-trader --since "-30 min" | grep -A3 "cycle exception"
```

No action usually needed — it self-recovers and posts a ✅. If it persists
past an hour during market hours: HALT flag + investigate; check
https://status.tradier.com.

## 6. Escalation cheatsheet

| Situation | Action |
|-----------|--------|
| Bot trading something it shouldn't | HALT flag, investigate with it paused |
| Position stuck, expiry today, short leg ITM | Stop bot, flatten that position manually (sec. 2) |
| Equity dropping fast, bot alive | Kill switch/breakers handle entries; flatten manually if exits aren't keeping up |
| Box unreachable | Tradier dashboard from anywhere: cancel orders, close positions |
| Not sure | HALT flag. It's free and reversible. |
