import math
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from arch import arch_model

from config import load_settings
from data import HistoricalStore, compute_log_returns
from features.garch import PCT_SCALE, fit_garch11
from model import GARCHBaseline, garch_forecast_path, per_horizon_metrics


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int = 0) -> pd.Series:
    template = arch_model(np.zeros(1), vol="Garch", p=1, q=1, mean="zero", rescale=False)
    np.random.seed(seed)
    sim = template.simulate(params=[omega, alpha, beta], nobs=n)
    return pd.Series(sim["data"].values / PCT_SCALE)


def test_closed_form_matches_arch_forecast():
    """Closed-form multi-step path must match arch's forecast(horizon=H) to 1e-6."""
    r = _simulate_garch(1500, 0.05, 0.10, 0.85, seed=11)
    fit = fit_garch11(r)

    # arch's multi-step variance forecast
    am = arch_model(r * PCT_SCALE, vol="Garch", p=1, q=1, mean="zero", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    horizon = 21
    arch_path_pct = res.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy()

    # Our closed-form path (in pct² before conversion)
    persistence = fit.alpha + fit.beta
    var_pct = fit.next_forecast_var
    manual_path_pct = [var_pct]
    for _ in range(horizon - 1):
        var_pct = fit.omega + persistence * var_pct
        manual_path_pct.append(var_pct)

    np.testing.assert_allclose(manual_path_pct, arch_path_pct, rtol=1e-6)
    print(f"closed_form_matches_arch_forecast: {horizon} steps match to 1e-6")

    # And the helper produces the same values
    helper_path = garch_forecast_path(fit, fit.next_forecast_var, horizon)
    expected_decimal = [v / (PCT_SCALE ** 2) for v in arch_path_pct]
    np.testing.assert_allclose(helper_path.daily_variances, expected_decimal, rtol=1e-6)
    print(f"  garch_forecast_path output matches arch (in decimal² units)")


def test_predict_forward_rv_near_unconditional():
    """For high-persistence GARCH, multi-step forecast pulls toward unconditional vol."""
    omega, alpha, beta = 0.05, 0.10, 0.85
    uncond_var_pct = omega / (1 - alpha - beta)        # in pct² units
    uncond_vol_decimal = math.sqrt(uncond_var_pct) / PCT_SCALE
    r = _simulate_garch(1500, omega, alpha, beta, seed=22)

    baseline = GARCHBaseline(min_history=252)
    pred = baseline.predict_forward_rv(r, horizon=21)
    rel = abs(pred - uncond_vol_decimal) / uncond_vol_decimal
    assert rel < 0.30, f"21-day pred {pred:.4f} vs uncond {uncond_vol_decimal:.4f} rel={rel:.3f}"
    print(f"predict_forward_rv: pred={pred:.5f} uncond={uncond_vol_decimal:.5f} rel_err={rel:.3%}")


def test_walk_forward_evaluate_shape_and_lookahead():
    n = 400
    r = _simulate_garch(n, 0.05, 0.10, 0.85, seed=33)
    horizons = (5, 10, 21)
    min_history = 100
    baseline = GARCHBaseline(refit_every=21, min_history=min_history)
    out = baseline.walk_forward_evaluate(r, horizons=horizons)

    assert out.shape == (n, 12), f"expected ({n}, 12), got {out.shape}"
    expected_cols = [
        "pred_rv_5", "actual_rv_5",
        "pred_rv_10", "actual_rv_10",
        "pred_rv_21", "actual_rv_21",
        "omega", "alpha", "beta", "resid_lb_pvalue", "aic", "bic",
    ]
    assert list(out.columns) == expected_cols

    # Look-ahead: last H rows of actual_rv_H must be NaN
    for h in horizons:
        col = f"actual_rv_{h}"
        last_h = out[col].iloc[-h:]
        assert last_h.isna().all(), f"{col}: last {h} rows should be NaN, got {last_h.notna().sum()} non-NaN"
        # Middle (post-warmup, pre-tail) rows must be non-NaN
        middle = out[col].iloc[min_history - 1 : -h]
        assert middle.notna().all(), f"{col}: middle has {middle.isna().sum()} NaN values"

    # Refits happened: omega has at least one non-NaN value
    assert out["omega"].notna().sum() >= n - min_history, "omega missing in too many post-warmup rows"

    # Diagnostic columns change exactly once per refit (~ceil((n - min_history+1)/21))
    n_active = n - (min_history - 1)
    expected_refits = (n_active + 20) // 21
    assert out["omega"].iloc[min_history - 1:].nunique() == expected_refits, (
        f"expected {expected_refits} distinct omega values, got {out['omega'].iloc[min_history - 1:].nunique()}"
    )
    print(f"walk_forward_evaluate: shape OK, look-ahead truncation OK, {expected_refits} refits")


def test_live_aapl_eval():
    """Live test using cached AAPL bars from step 3."""
    settings = load_settings()
    store = HistoricalStore(settings.cache_db_path)
    try:
        end = date.today()
        while end.weekday() >= 5:
            end -= timedelta(days=1)
        start = end - timedelta(days=730)

        bars = store.get_bars("AAPL", start, end)
        assert not bars.empty, "no cached AAPL bars — run step 3 tests first"
        returns = compute_log_returns(bars["close"])
        print(f"  AAPL: {len(bars)} bars, {len(returns)} returns")

        baseline = GARCHBaseline(refit_every=21, min_history=100)
        t0 = time.monotonic()
        eval_df = baseline.walk_forward_evaluate(returns, horizons=(5, 10, 21))
        elapsed = time.monotonic() - t0
        print(f"  walk_forward_evaluate (1 ticker, 2y): {elapsed:.2f}s")
        assert elapsed < 15.0, f"FAIL: walk-forward eval took {elapsed:.2f}s, must be <15s"

        metrics = per_horizon_metrics(eval_df, horizons=(5, 10, 21))
        print("\nGARCH baseline AAPL eval (2y):")
        print(metrics.to_string())

        for h in (5, 10, 21):
            row = metrics.loc[h]
            assert not math.isnan(row["r2"]), f"R² is NaN at horizon {h}"
            assert row["rmse"] < 0.05, f"RMSE {row['rmse']} too high at horizon {h}"
            assert row["n"] > 100, f"only {row['n']} pairs at horizon {h}"
    finally:
        store.close()


def main() -> int:
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    test_closed_form_matches_arch_forecast()
    test_predict_forward_rv_near_unconditional()
    test_walk_forward_evaluate_shape_and_lookahead()
    test_live_aapl_eval()
    print("\nall garch_baseline tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
