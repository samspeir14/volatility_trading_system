# options-trader

A short-dated options volatility bot built around one edge: a machine-learned
forecast of **tomorrow's realized volatility**, traded in the options where a
1-day timing signal still matters — expirations **1 to 14 days out**.

Live since 2026-08-11 (Tradier sandbox as the paper track record + a small
real-money profile), running autonomously on EC2 under systemd.

## The strategy

1. **Forecast.** A pooled LightGBM predicts each ticker's next-day
   log-GK-volatility *deviation* from its own 63-day baseline (within-stock
   timing skill, not "TSLA is more volatile than KO"). A HAR-RV benchmark
   trains alongside it; a weekly acceptance gate (out-of-sample within-ticker
   deviation R² — the strategy's singular metric) routes production to
   whichever is actually better: LightGBM, HAR, or their 50/50 blend
   (`route=lgbm|har|blend`).
2. **Term-project.** The 1-day forecast is decayed toward the ticker's mean
   vol along its GARCH persistence to the exact DTE of each candidate option,
   then compared to that expiration's ATM implied vol. Beyond ~2 weeks the
   deviation has decayed away — which is why the bot only trades DTE 1-14
   (`MIN_ENTRY_DTE`/`MAX_ENTRY_DTE`).
3. **Gate.** A tenor-matched VRP z-score sets direction only when the IV
   premium is extreme vs the ticker's own history (SELL at z ≥ +1.5, BUY at
   z ≤ −1.25), then the candidate must clear the full ladder: VIX
   term-structure crash veto, model-agreement, 0.25 divergence cap, earnings
   and macro blocks over the position's whole life, index-ETF long-straddle
   exclusion, per-leg liquidity, credit floors, and a cost gate (edge must
   cover 2× spread+fees).
4. **Trade.** BUY → long ATM straddle. SELL → iron condor (ATM body, ~1σ
   wings priced off the model's own forecast). Orders walk a price ladder
   from mid; every fill logs slippage (`grep TCA`).
5. **Manage.** Exits run every cycle regardless of entry guards: thesis
   reversal, profit target / stop, fail-closed earnings exit, assignment-risk
   close-outs, and expiry handling built for deliberately short-dated entries
   (a 1-2 DTE position rides to expiration day and closes in the final 2
   hours before the bell).

Risk sits outside the strategy: per-trade / per-ticker / sector / portfolio
caps, Greek limits, buying-power buffer, a daily-loss kill switch, weekly and
monthly drawdown breakers, a bars-freshness guard, a manual HALT flag, and a
cron-side dead-man switch.

## Setup

```bash
git clone <this repo>
cd options-trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in TRADIER_API_KEY and TRADIER_ACCOUNT_ID
python -m tests.test_tradier_client   # sandbox smoke test
python -m main --once                 # one full cycle, then exit
```

`ACCOUNT_PROFILE=standard` (~$100k calibration) or `small` (~$10k: cheap-name
watchlist, retuned caps, 1-lot orders). Real-money trading additionally
requires `TRADIER_ENV=production` **and** `TRADIER_LIVE_TRADING_CONFIRMED=YES`.

Deployment, the Sunday retrain cron, the nightly metrics reconcile, and the
operational guards are documented in [deploy/README.md](deploy/README.md);
incident procedures in [deploy/RUNBOOK.md](deploy/RUNBOOK.md); open research
questions in [experiments/RESEARCH_BACKLOG.md](experiments/RESEARCH_BACKLOG.md).

## Architecture

`config/` settings + watchlists · `data/` async Tradier client, market/chain
scans, bars cache, earnings + macro calendars · `features/` vol estimators,
technicals, cross-ticker context, h=1 target · `model/` LightGBM + HAR h=1
deviation predictors, GARCH term projection, retrain evaluation · `signals/`
the gate ladder + VRP gap history · `execution/` order manager (ladder, dedup,
TCA) + order log · `positions/` tracking, reconciliation, exit manager ·
`risk/` pre-trade gates, kill switch, drawdown breakers, ops guards · `logs/`
structured logging + daily Slack summary · `main.py` the 5-minute cycle
orchestrator · `tests/` unit + live sandbox tests (the retrain job doubles as
`tests/test_model_retraining.py`) · `experiments/` the h=1 model lab and
research artifacts.

## History

The bot began as a VRP-harvest premium seller with multi-horizon (5/10/21-day)
vol models and longer-dated structures. Seven years of research said the
premium *level* was not timeable at any tenor — the harvest strategy was
retired 2026-08 and the multi-horizon pipeline deleted; git history preserves
both. What survived is the one thing that showed real skill: predicting the
next day's volatility, expressed in the shortest-dated options.
