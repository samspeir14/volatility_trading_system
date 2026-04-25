import sys

from config import load_settings
from data.tradier_client import TradierAPIError, TradierClient


def main() -> int:
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run smoke test against env={settings.env!r}", file=sys.stderr)
        return 2

    client = TradierClient(settings)

    try:
        profile = client.get_profile()
        balances = client.get_balances()
        aapl = client.get_quote("AAPL")
    except TradierAPIError as e:
        print(f"Tradier API error: {e}", file=sys.stderr)
        return 1

    assert profile.get("name"), f"profile.name missing: {profile}"
    assert "total_cash" in balances, f"balances missing total_cash: {balances}"
    assert balances["account_number"] == settings.account_id, (
        f"account mismatch: settings={settings.account_id} balances={balances['account_number']}"
    )
    for field in ("last", "bid", "ask"):
        assert field in aapl, f"AAPL quote missing {field!r}: {aapl}"

    acct = profile["account"]
    print(f"User:    {profile['name']} (id={profile['id']})")
    print(f"Account: {acct['account_number']} "
          f"(type={acct['type']}, option_level={acct['option_level']})")
    print(f"Cash:    ${balances['total_cash']:,.2f}")
    print(f"Equity:  ${balances['total_equity']:,.2f}")
    margin = balances.get("margin", {})
    print(f"BP:      stocks=${margin.get('stock_buying_power', 0):,.2f} "
          f"options=${margin.get('option_buying_power', 0):,.2f}")
    print(f"AAPL:    last=${aapl['last']} bid=${aapl['bid']} ask=${aapl['ask']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
