# Research backlog

Strategy experiments worth running BEFORE changing what's deployed. The
sequence is always: study offline (vrp_log / order log / replay) → compare
against the deployed baseline → only then consider a live change. Ordered by
expected value.

The h=1 pivot (2026-08-11) reset this list: the bot now trades only the
next-day vol forecast in DTE 1-14 options. Items are scoped to that book.

## 1. Edge decay by DTE bucket (does 14 earn its place?)

The entry window is 1-14 DTE, but the h=1 forecast is mostly GARCH decay by
the back of that window — the model's skill concentrates at the front. Once
enough round trips accumulate: realized P&L, credit capture, and win rate by
DTE bucket (1-2, 3-5, 6-9, 10-14) from the order log + divergence history.
If the back buckets don't pay, tighten `MAX_ENTRY_DTE` (env-tunable, no code
change). Do not pre-judge: the back buckets also carry less gamma risk per
trade.

## 2. Finer VRP bands inside 0-14

The z-gate compares each candidate against gap history from its DTE band —
currently one band for the whole tradeable window. The gap is fattest at the
shortest tenors, so a 2-DTE candidate scored against mostly-10-DTE history
may read systematically rich. Offline study on `vrp_log` (raw dte is stored
per row; re-banding is query-time only): recompute z-scores under
(0,2)/(3,7)/(8,14) and compare gate decisions + subsequent realized gaps.
Only re-band live after the short buckets have enough rows (they only started
accruing at the 2026-08 pivot; at VRP_MIN_OBS=60 a (0,2) band would stay
silent until ~November).

## 3. Profit-target / stop sweep for the 1-14 DTE book

Current management: condor +50% of credit PT / −100% stop; straddle +100% /
−50%; deliberately short-dated entries ride to an expiry-day close. These
numbers predate the short-DTE pivot. For ATM-body structures at DTE ≤ 8 the
common practice is earlier profit-taking (~25%) because post-50% the
remaining theta is small against the gamma still on the table. Sweep PT ∈
{25%, 35%, 50%, none} × stop ∈ {−50%, −100%, none} on replay/paper fills
before touching the live numbers.

## 4. Weekend and event-day effects in 1-3 DTE IV

A Friday 1-DTE (Monday-expiry) option prices two calendar days of no trading;
the tenor-matched VRP gap g = log(IV) − log(realized) may need a
calendar-vs-trading-day adjustment at the very front of the curve, or the
z-history absorbs it. Check: distribution of g by weekday at DTE 1-3 in
`vrp_log`. If Friday g is structurally fatter, the gate is selling weekend
decay, not mispricing.

## 5. Gamma-cap recalibration for CALIBRATION_STANDARD

The standard (paper) profile's 1% portfolio gamma cap vs 5% vega cap reflects
the old longer-dated vega-driven book; raw ATM gamma scales ~1/(S·σ·√T), so a
1-14 DTE ladder hits the gamma cap much earlier (the small profile already
runs 10% for exactly this reason). Evidence first: count gamma-cap rejections
in `risk_rejection_log` over a few weeks of short-DTE trading, then set the
paper cap from data. Do not touch the live-money profile without that
evidence.

## 6. Delta-targeted short strikes (vs the ATM body) — carried over

The deployed SELL structure sells both shorts at the entry ATM strike
(structurally an iron butterfly) with ~1σ wings — maximum premium, maximum
gamma, a short essentially always ITM near expiry. The mainstream places
shorts by DELTA (~25-30Δ shorts, ~10Δ wings): smile-aware, higher expiry win
rate, less credit, much less gamma. Replay over the short-DTE entries with
shorts at {ATM, 30Δ, 25Δ} × wings at {1σ, 10Δ}; judge P&L per dollar of
worst-case risk, not win rate. Interacts strongly with #3 — run as a grid.

## 7. Beta-weighted portfolio delta cap — carried over

The RiskManager delta cap sums raw per-name delta dollars; beta-weighting to
SPY stops offsetting deltas in correlated names from pretending to hedge each
other. Needs per-name betas (trailing 252d regression against SPY — bars are
cached). Low urgency while orders are 1-lot; matters when sizing scales.

## 8. Execution report (TCA + credit capture) — carried over

The plumbing exists: `arrival_mid` on every submission and close attempt,
realized P&L per position, entry IV and realized vol in the divergence
history. Formalize the weekly read: credit capture % vs modeled, win rate,
slippage vs arrival mid by symbol and by DTE bucket, veto counts by gate. A
`scripts/execution_report.py` that prints the table is enough — resist the
dashboard until the numbers earn it.

## 9. Written promotion gate for the small profile — carried over

Decide thresholds BEFORE looking at results: e.g. "scale live sizing only
after ≥N filled round trips with credit capture ≥ X%, realized slippage ≤ Y%
of mid, and zero assignment-risk incidents". Write the numbers into
deploy/README §6d when chosen.

---

### Retired 2026-08 (harvest era — kept for the record)

The delta-targeted-strikes and PT/stop-sweep items above originated as
harvest-structure experiments; they carry over because the structure did.
Everything else harvest-specific (fly-structure credit studies against the
DoltHub IV history, harvest promotion gates) died with the strategy — the
7-year verdict was that the premium level is not timeable at any tenor,
formula or ML. See `experiments/spread_history.py` / `dolthub_iv_pull.py`
for how that IV dataset was built; git history has the retired analyses.
