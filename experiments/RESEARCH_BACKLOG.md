# Research backlog

Strategy experiments worth running BEFORE changing what's deployed. Each of
these changes the win-rate/credit profile away from what the 2026-07 research
validated, so the sequence is: backtest with the spread-history harness →
compare against the deployed baseline → only then consider a live change.
Ordered by expected value.

## 1. Delta-targeted short strikes (vs the ATM body)

The deployed harvest structure sells BOTH short legs at the entry ATM strike
(structurally an iron butterfly, though the code says condor) with ~1σ wings —
maximum premium, maximum gamma, and a short leg that is essentially always ITM
near expiry. The systematic-premium-selling mainstream places shorts by DELTA
instead (e.g. ~25–30Δ shorts with ~10Δ wings): smile-aware (the put side sits
wider in %-terms, matching skew), higher expiry win rate, less credit per
trade, much less gamma in the fat zone.

Experiment: replay the DTE 5–15 entry rule over the DoltHub IV history with
short strikes at {ATM, 30Δ, 25Δ, 16Δ} × wings at {1σ, 10Δ}. Compare credit
capture, win rate, max-loss frequency, and P&L per unit of wing risk. The
question is not "which wins more" (OTM always wins more often) but which earns
more per dollar of worst-case risk after the profit-target management below.

## 2. Profit-target / stop sweep for the fly structure

Current management: +50% of credit PT, −100% stop, ride toward expiry
otherwise. For ATM-body structures the common practice is earlier profit
taking (~25%) because post-50% the remaining theta is small against the gamma
still on the table at DTE ≤ 8. Sweep PT ∈ {25%, 35%, 50%, none} × stop ∈
{−50%, −100%, −200%, none} on the same replay. Interacts strongly with #1 —
run as a grid, not sequentially.

## 3. Beta-weighted portfolio delta cap

The RiskManager delta cap sums raw per-name delta dollars; industry practice
beta-weights to SPY so that offsetting deltas in correlated names don't
pretend to hedge each other. Needs per-name betas (trailing 252d regression
against SPY — the bars are already cached). Low urgency while orders are
1-lot condors, matters when sizing scales.

## 4. Expected-vs-realized dashboard (TCA + credit capture)

The plumbing exists as of 2026-07: `arrival_mid` on every submission and
close attempt, realized P&L per position, entry IV and trail-63 RV in the
divergence history. Formalize the weekly read: credit capture % vs modeled,
win rate vs the 72% benchmark, slippage vs arrival mid by symbol, veto
counts by type. A `scripts/execution_report.py` that prints the table from
the order log is enough — resist the dashboard until the numbers earn it.

## 5. Written promotion gate for small_harvest going live

Decide the thresholds BEFORE looking at results (the whole point): e.g.
"go live only after ≥N filled round trips with credit capture ≥ X%, realized
slippage ≤ Y% of mid, and zero assignment-risk incidents". Write the numbers
down in deploy/README §6d when chosen.
