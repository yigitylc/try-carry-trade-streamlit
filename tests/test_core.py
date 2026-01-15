import math
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dashboard_core import (
    TryHolderParams,
    UsdHolderParams,
    CarryTradeModel,
    compute_try_holder_metrics,
    compute_usd_holder_metrics,
)


def test_interest_math_try_holder():
    params = TryHolderParams(
        principal_try=1_000_000,
        entry_bank_rate_try_to_usd=40.0,
        spot_entry=39.5,
        start_date=datetime(2025, 1, 1),
        term_days_calendar=30,
        deposit_rate_annual=0.40,
        stopaj_rate=0.10,
        swift_fee_usd=0.0,
        usd_rf_rate_annual=0.03,
        exit_spread=0.01,
    )
    metrics = compute_try_holder_metrics(params)

    expected_gross = params.principal_try * params.deposit_rate_annual * (params.term_days_calendar / 365)
    expected_net = expected_gross * (1 - params.stopaj_rate)
    expected_final = params.principal_try + expected_net

    assert math.isclose(metrics.gross_interest, expected_gross, rel_tol=1e-9)
    assert math.isclose(metrics.net_interest, expected_net, rel_tol=1e-9)
    assert math.isclose(metrics.final_try, expected_final, rel_tol=1e-9)


def test_break_even_math():
    params = TryHolderParams(
        principal_try=1_000_000,
        entry_bank_rate_try_to_usd=40.0,
        spot_entry=39.5,
        start_date=datetime(2025, 1, 1),
        term_days_calendar=30,
        deposit_rate_annual=0.40,
        stopaj_rate=0.10,
        swift_fee_usd=10.0,
        usd_rf_rate_annual=0.03,
        exit_spread=0.01,
    )
    metrics = compute_try_holder_metrics(params)
    model = CarryTradeModel(metrics)

    outcome = model.compute_outcome(metrics.spot_be)
    assert abs(outcome.excess_ret_vs_tbill) < 0.01


def test_principal_scaling_swift_fee_effect():
    params = TryHolderParams(
        principal_try=1_000_000,
        entry_bank_rate_try_to_usd=40.0,
        spot_entry=39.5,
        start_date=datetime(2025, 1, 1),
        term_days_calendar=30,
        deposit_rate_annual=0.40,
        stopaj_rate=0.10,
        swift_fee_usd=0.0,
        usd_rf_rate_annual=0.03,
        exit_spread=0.01,
    )
    metrics = compute_try_holder_metrics(params)
    metrics_scaled = compute_try_holder_metrics(
        TryHolderParams(**{**params.__dict__, "principal_try": params.principal_try * 2})
    )

    base_outcome = CarryTradeModel(metrics).compute_outcome(metrics.spot_entry)
    scaled_outcome = CarryTradeModel(metrics_scaled).compute_outcome(metrics_scaled.spot_entry)

    assert math.isclose(base_outcome.ret_vs_baseline, scaled_outcome.ret_vs_baseline, rel_tol=1e-9)

    params_swift = TryHolderParams(**{**params.__dict__, "swift_fee_usd": 25.0})
    metrics_swift = compute_try_holder_metrics(params_swift)
    metrics_swift_scaled = compute_try_holder_metrics(
        TryHolderParams(**{**params_swift.__dict__, "principal_try": params_swift.principal_try * 2})
    )

    base_swift = CarryTradeModel(metrics_swift).compute_outcome(metrics_swift.spot_entry)
    scaled_swift = CarryTradeModel(metrics_swift_scaled).compute_outcome(metrics_swift_scaled.spot_entry)

    assert not math.isclose(base_swift.ret_vs_baseline, scaled_swift.ret_vs_baseline, rel_tol=1e-9)


def test_usd_holder_entry_conversion_uses_bank_rate():
    params = UsdHolderParams(
        principal_usd=50_000,
        entry_bank_rate_usd_to_try=42.0,
        spot_entry=43.0,
        start_date=datetime(2025, 1, 1),
        term_days_calendar=30,
        deposit_rate_annual=0.40,
        stopaj_rate=0.10,
        swift_fee_usd=0.0,
        usd_rf_rate_annual=0.03,
        exit_spread=0.01,
    )
    metrics = compute_usd_holder_metrics(params)
    assert math.isclose(metrics.principal_try, params.principal_usd * params.entry_bank_rate_usd_to_try, rel_tol=1e-9)


def test_try_holder_baseline_uses_entry_bank_rate():
    params = TryHolderParams(
        principal_try=1_000_000,
        entry_bank_rate_try_to_usd=41.0,
        spot_entry=39.5,
        start_date=datetime(2025, 1, 1),
        term_days_calendar=30,
        deposit_rate_annual=0.40,
        stopaj_rate=0.10,
        swift_fee_usd=0.0,
        usd_rf_rate_annual=0.03,
        exit_spread=0.01,
    )
    metrics = compute_try_holder_metrics(params)
    assert math.isclose(metrics.usd0_baseline, params.principal_try / params.entry_bank_rate_try_to_usd, rel_tol=1e-9)
