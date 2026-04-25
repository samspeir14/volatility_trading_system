# options-trader

Options volatility-arbitrage bot. Compares ML-predicted realized volatility against live implied volatility on a ~20-ticker watchlist and trades the divergence via the Tradier API.

Currently at **Step 1** of a 10-step build (see project spec). Only the config and Tradier client are implemented; the rest are stubs.

## Setup

```bash
git clone git@github.com:<user>/options-trader.git
cd options-trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in TRADIER_API_KEY and TRADIER_ACCOUNT_ID
python -m tests.test_tradier_client
```

The smoke test hits the Tradier sandbox and prints your account profile, balances, and an AAPL quote. Switching to live trading is a single env-var change: `TRADIER_ENV=production` plus a production API token.

## Architecture (planned)

`config/` settings + watchlist · `data/` Tradier client + market data + historical bars · `features/` realized-vol + technicals + cross-ticker · `model/` GARCH baseline + XGBoost · `signals/` IV-vs-predicted-RV divergence · `execution/` order placement · `positions/` tracking + exits · `risk/` pre-trade checks + kill switch · `logs/` structured logging · `main.py` orchestrator.
