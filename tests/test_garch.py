import math
import sys

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, Normal

from features.garch import PCT_SCALE, fit_garch11, garch_features_walk_forward


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int = 0) -> pd.Series:
    """Simulate a synthetic GARCH(1,1) series of pct returns (zero mean)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    r[0] = math.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
        r[t] = math.sqrt(sigma2[t]) * z[t]
    return pd.Series(r / PCT_SCALE)  # convert pct → decimal returns


def test_parameter_recovery():
    true_omega = 0.05
    true_alpha = 0.10
    true_beta = 0.85
    r = _simulate_garch(2000, true_omega, true_alpha, true_beta, seed=42)
    fit = fit_garch11(r)
    # 30% relative tolerance per plan
    for name, true_v, fit_v in [
        ("omega", true_omega, fit.omega),
        ("alpha", true_alpha, fit.alpha),
        ("beta", true_beta, fit.beta),
    ]:
        rel = abs(fit_v - true_v) / true_v
        assert rel < 0.30, f"{name}: true={true_v} fit={fit_v:.4f} rel_err={rel:.3f}"
        print(f"  {name}: true={true_v} fit={fit_v:.4f} rel_err={rel:.3%}")
    print("parameter_recovery: OK")


def test_daily_recursion_correctness():
    """Walk-forward with refit_every=999 → after the single fit, every row must
    follow the closed-form GARCH recursion exactly."""
    r = _simulate_garch(400, 0.05, 0.10, 0.85, seed=7)
    out = garch_features_walk_forward(r, refit_every=999, min_history=200)

    # Get the same fit the loop would have produced at i=199
    fit_ref = fit_garch11(r.iloc[:200])

    # Row 199 records σ²_{200} = fit.next_forecast_var (in pct² → divide by 1e4)
    expected_at_199 = fit_ref.next_forecast_var / (PCT_SCALE ** 2)
    actual_at_199 = out["garch_forecast_var"].iloc[199]
    assert math.isclose(expected_at_199, actual_at_199, rel_tol=1e-9), (
        f"row 199: expected {expected_at_199} got {actual_at_199}"
    )

    # Rows 200+ follow the daily recursion using fit_ref's params
    var_pct_sq = fit_ref.next_forecast_var
    for i in range(200, len(r)):
        r_pct = float(r.iloc[i]) * PCT_SCALE
        var_pct_sq = fit_ref.omega + fit_ref.alpha * (r_pct ** 2) + fit_ref.beta * var_pct_sq
        expected = var_pct_sq / (PCT_SCALE ** 2)
        actual = out["garch_forecast_var"].iloc[i]
        assert math.isclose(expected, actual, rel_tol=1e-9), (
            f"row {i}: expected {expected} got {actual}"
        )
    print(f"daily_recursion_correctness: {len(r) - 200} rows match closed-form recursion to 1e-9")


def test_periodic_refit_with_daily_updates():
    r = _simulate_garch(400, 0.05, 0.10, 0.85, seed=13)
    min_history = 100
    refit_every = 21
    out = garch_features_walk_forward(r, refit_every=refit_every, min_history=min_history)

    fcast = out["garch_forecast_var"].iloc[min_history - 1:].dropna()
    pval = out["garch_resid_lb_pvalue"].iloc[min_history - 1:].dropna()

    # (a) p-value changes once per refit. Number of refits = ceil((n - (min_history-1)) / refit_every)
    n_active = len(r) - (min_history - 1)
    expected_refits = (n_active + refit_every - 1) // refit_every
    n_unique_pvals = pval.nunique()
    assert n_unique_pvals == expected_refits, (
        f"expected {expected_refits} distinct LB p-values, got {n_unique_pvals}"
    )
    print(f"periodic_refit: {n_unique_pvals} distinct LB p-values across {n_active} rows "
          f"(expected {expected_refits} refits at every {refit_every})")

    # (b) forecast changes on every row (daily recursion firing, not frozen between refits)
    diffs = fcast.diff().dropna()
    assert (diffs.abs() > 0).all(), (
        f"forecast was constant on {(diffs == 0).sum()} consecutive rows — daily recursion broken"
    )
    print(f"daily_updates: forecast changed on every one of {len(diffs)} consecutive rows")


def main() -> int:
    test_parameter_recovery()
    test_daily_recursion_correctness()
    test_periodic_refit_with_daily_updates()
    print("all garch tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
