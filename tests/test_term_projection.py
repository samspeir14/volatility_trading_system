"""Tests for level reconstruction and GARCH-persistence term projection."""
import math
import sys

from model.term_structure import (
    PHI_MAX,
    PHI_MIN,
    TRADING_DAYS_PER_YEAR,
    project_term_vol,
    reconstruct_level,
)

B = -4.6      # baseline log daily vol (~1% daily)
DEV = 0.4     # elevated deviation


def test_reconstruct_level():
    daily, annual = reconstruct_level(B, DEV)
    assert math.isclose(daily, math.exp(B + DEV), rel_tol=1e-12)
    assert math.isclose(annual, daily * math.sqrt(TRADING_DAYS_PER_YEAR), rel_tol=1e-12)
    print("reconstruct: daily=exp(b+dev), annual=daily*sqrt(252)")


def test_k1_equals_level_forecast():
    # 1 calendar day → K=1: forecast is just the annualized day-1 vol
    expected = math.exp(B + DEV) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert math.isclose(project_term_vol(B, DEV, 0.9, 1), expected, rel_tol=1e-12)
    print("k1: matches annualized exp(b+dev)")


def test_long_dte_converges_to_baseline():
    baseline_ann = math.exp(B) * math.sqrt(TRADING_DAYS_PER_YEAR)
    day1_ann = math.exp(B + DEV) * math.sqrt(TRADING_DAYS_PER_YEAR)
    far = project_term_vol(B, DEV, 0.85, 3650)  # 10 years out
    assert abs(far - baseline_ann) < abs(day1_ann - baseline_ann) * 0.01
    assert math.isclose(far, baseline_ann, rel_tol=0.02)
    print(f"convergence: 10y forecast {far:.4f} ≈ baseline {baseline_ann:.4f}")


def test_monotone_between_day1_and_baseline():
    day1 = project_term_vol(B, DEV, 0.9, 1)
    prev = day1
    for dte in (5, 10, 21, 45, 90):
        cur = project_term_vol(B, DEV, 0.9, dte)
        assert cur < prev, f"forecast should decay toward baseline (dte={dte})"
        prev = cur
    baseline_ann = math.exp(B) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert prev > baseline_ann  # never undershoots for positive dev
    # negative deviation: decays upward toward baseline
    low1 = project_term_vol(B, -DEV, 0.9, 1)
    low45 = project_term_vol(B, -DEV, 0.9, 45)
    assert low1 < low45 < baseline_ann
    print("monotone: decays toward baseline from both sides")


def test_phi_clipped_both_ends():
    # phi below the floor behaves exactly like the floor; above like the cap
    assert math.isclose(
        project_term_vol(B, DEV, 0.10, 30),
        project_term_vol(B, DEV, PHI_MIN, 30),
        rel_tol=1e-12,
    )
    assert math.isclose(
        project_term_vol(B, DEV, 1.50, 30),
        project_term_vol(B, DEV, PHI_MAX, 30),
        rel_tol=1e-12,
    )
    print(f"phi_clip: [{PHI_MIN}, {PHI_MAX}] enforced")


def test_closed_form_matches_naive_sum():
    phi, dte = 0.93, 30
    k_days = max(1, round(dte * 252 / 365))
    var_base = math.exp(2 * B)
    var_1 = math.exp(2 * (B + DEV))
    naive = sum(
        var_base + (var_1 - var_base) * phi ** (k - 1) for k in range(1, k_days + 1)
    ) / k_days
    expected = math.sqrt(naive) * math.sqrt(252)
    assert math.isclose(project_term_vol(B, DEV, phi, dte), expected, rel_tol=1e-12)
    print("closed_form: matches explicit var_k sum")


def test_invalid_dte_raises():
    try:
        project_term_vol(B, DEV, 0.9, 0)
    except ValueError:
        print("invalid_dte: raises on dte < 1")
        return
    raise AssertionError("expected ValueError for dte=0")


def main() -> int:
    test_reconstruct_level()
    test_k1_equals_level_forecast()
    test_long_dte_converges_to_baseline()
    test_monotone_between_day1_and_baseline()
    test_phi_clipped_both_ends()
    test_closed_form_matches_naive_sum()
    test_invalid_dte_raises()
    print("all term_projection tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
