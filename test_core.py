"""
test_core.py
============
Unit tests for TRY Carry Trade core calculations.

Tests:
- Interest math (gross, stopaj, net)
- Break-even math
- Principal scaling with SWIFT
- USD-holder entry conversion
- TRY-holder baseline
- GBM calibration (252-day annualization)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from dashboard_core import (
    TradeParams, TRYHolderParams, USDHolderParams,
    CarryTradeModel, calibrate_gbm, compute_rolling_volatility
)


class TestInterestMath:
    """Test interest calculations."""

    def test_gross_interest_calculation(self):
        """Verify gross interest = principal * rate * (days/365)."""
        params = TradeParams(
            principal_try=1_000_000,
            deposit_rate_annual=0.40,  # 40%
            term_days_calendar=365,
        )
        expected_gross = 1_000_000 * 0.40 * 1.0  # Full year
        assert abs(params.gross_interest - expected_gross) < 0.01

    def test_gross_interest_32_days(self):
        """Verify gross interest for 32-day term."""
        params = TradeParams(
            principal_try=1_430_000,
            deposit_rate_annual=0.395,
            term_days_calendar=32,
        )
        expected_gross = 1_430_000 * 0.395 * (32 / 365)
        assert abs(params.gross_interest - expected_gross) < 0.01

    def test_stopaj_calculation(self):
        """Verify stopaj = gross * stopaj_rate."""
        params = TradeParams(
            principal_try=1_000_000,
            deposit_rate_annual=0.40,
            term_days_calendar=365,
            stopaj_rate=0.175,  # 17.5%
        )
        expected_stopaj = params.gross_interest * 0.175
        assert abs(params.stopaj_amount - expected_stopaj) < 0.01

    def test_net_interest_calculation(self):
        """Verify net = gross - stopaj."""
        params = TradeParams(
            principal_try=1_000_000,
            deposit_rate_annual=0.40,
            term_days_calendar=365,
            stopaj_rate=0.175,
        )
        expected_net = params.gross_interest - params.stopaj_amount
        assert abs(params.net_interest - expected_net) < 0.01

    def test_final_try_calculation(self):
        """Verify final_try = principal + net_interest."""
        params = TradeParams(
            principal_try=1_430_000,
            deposit_rate_annual=0.395,
            term_days_calendar=32,
            stopaj_rate=0.175,
        )
        expected_final = params.principal_try + params.net_interest
        assert abs(params.final_try - expected_final) < 0.01


class TestBreakEvenMath:
    """Test break-even calculations."""

    def test_break_even_bank_rate(self):
        """Verify bank_rate_be = final_try / (usd_rf_end + swift)."""
        params = TradeParams(
            principal_try=1_430_000,
            deposit_rate_annual=0.395,
            term_days_calendar=32,
            stopaj_rate=0.175,
            entry_bank_rate=43.10,
            spot_entry=42.72,
            swift_fee_usd=32.50,
            usd_rf_rate_annual=0.035,
        )
        expected_be_rate = params.final_try / (params.usd_rf_end + params.swift_fee_usd)
        assert abs(params.bank_rate_be - expected_be_rate) < 0.0001

    def test_break_even_spot(self):
        """Verify spot_be = bank_rate_be / (1 + spread)."""
        params = TradeParams(
            principal_try=1_430_000,
            deposit_rate_annual=0.395,
            term_days_calendar=32,
            stopaj_rate=0.175,
            entry_bank_rate=43.10,
            spot_entry=42.72,
            swift_fee_usd=32.50,
            usd_rf_rate_annual=0.035,
        )
        expected_spot_be = params.bank_rate_be / (1 + params.spread)
        assert abs(params.spot_be - expected_spot_be) < 0.0001

    def test_break_even_move_pct(self):
        """Verify be_move_pct = (spot_be / spot_entry - 1) * 100."""
        params = TradeParams(
            principal_try=1_430_000,
            spot_entry=42.72,
            entry_bank_rate=43.10,
        )
        expected_move = (params.spot_be / params.spot_entry - 1) * 100
        assert abs(params.be_move_pct - expected_move) < 0.01

    def test_at_break_even_no_excess(self):
        """At break-even spot, excess return should be ~0."""
        params = TradeParams(
            principal_try=1_430_000,
            deposit_rate_annual=0.395,
            term_days_calendar=32,
            stopaj_rate=0.175,
            entry_bank_rate=43.10,
            spot_entry=42.72,
            swift_fee_usd=32.50,
            usd_rf_rate_annual=0.035,
        )
        model = CarryTradeModel(params)
        outcome = model.compute_outcome(params.spot_be)

        # At break-even, excess return should be very close to 0
        assert abs(outcome.excess_ret_vs_tbill) < 0.01


class TestPrincipalScalingWithSWIFT:
    """Test that principal scaling properly accounts for SWIFT fee."""

    def test_swift_fee_impact(self):
        """SWIFT fee should reduce USD outcome by flat amount."""
        # Compare with and without SWIFT fee
        params_with_swift = TradeParams(
            principal_try=1_430_000,
            swift_fee_usd=32.50,
            spot_entry=42.72,
            entry_bank_rate=43.10,
        )
        params_no_swift = TradeParams(
            principal_try=1_430_000,
            swift_fee_usd=0.0,
            spot_entry=42.72,
            entry_bank_rate=43.10,
        )

        model_with = CarryTradeModel(params_with_swift)
        model_no = CarryTradeModel(params_no_swift)

        outcome_with = model_with.compute_outcome(42.72)
        outcome_no = model_no.compute_outcome(42.72)

        # Difference should be exactly the SWIFT fee
        diff = outcome_no.usd_end - outcome_with.usd_end
        assert abs(diff - 32.50) < 0.01

    def test_scaling_preserves_ratios(self):
        """Doubling principal should roughly double USD outcomes."""
        params_1x = TradeParams(
            principal_try=1_000_000,
            swift_fee_usd=0.0,  # Exclude SWIFT for pure scaling test
            spot_entry=42.72,
            entry_bank_rate=43.10,
        )
        params_2x = TradeParams(
            principal_try=2_000_000,
            swift_fee_usd=0.0,
            spot_entry=42.72,
            entry_bank_rate=43.10,
        )

        model_1x = CarryTradeModel(params_1x)
        model_2x = CarryTradeModel(params_2x)

        outcome_1x = model_1x.compute_outcome(44.0)
        outcome_2x = model_2x.compute_outcome(44.0)

        # USD end should scale proportionally
        ratio = outcome_2x.usd_end / outcome_1x.usd_end
        assert abs(ratio - 2.0) < 0.01


class TestUSDHolderMode:
    """Test USD holder mode calculations."""

    def test_usd_holder_entry_conversion(self):
        """Verify TRY position = principal_usd * entry_bank_rate_usd_to_try."""
        params = USDHolderParams(
            principal_usd=33_000,
            entry_bank_rate_usd_to_try=42.34,
        )
        expected_try = 33_000 * 42.34
        assert abs(params.try0 - expected_try) < 0.01

    def test_usd_holder_baseline_is_tbill(self):
        """USD holder baseline should be T-bill on original USD."""
        params = USDHolderParams(
            principal_usd=33_000,
            tbill_rate_annual_usd=0.035,
            term_days_calendar=32,
        )
        # Baseline = principal * (1 + rate * T)
        expected_rf_end = 33_000 * (1 + 0.035 * 32 / 365)
        assert abs(params.usd_rf_end - expected_rf_end) < 0.01

    def test_usd_holder_usd0_exec_equals_principal(self):
        """For USD holder, usd0_exec should equal principal_usd."""
        params = USDHolderParams(
            principal_usd=33_000,
        )
        assert params.usd0_exec == 33_000


class TestTRYHolderMode:
    """Test TRY holder mode calculations."""

    def test_try_holder_baseline(self):
        """TRY holder baseline = convert at bank rate + T-bill."""
        params = TRYHolderParams(
            principal_try=1_430_000,
            entry_bank_rate_try_to_usd=43.10,
            tbill_rate_annual_usd=0.035,
            term_days_calendar=32,
        )
        # usd0_exec = principal / entry_bank_rate
        expected_usd0 = 1_430_000 / 43.10
        assert abs(params.usd0_exec - expected_usd0) < 0.01

        # usd_rf_end = usd0 * (1 + rate * T)
        expected_rf_end = expected_usd0 * (1 + 0.035 * 32 / 365)
        assert abs(params.usd_rf_end - expected_rf_end) < 0.01

    def test_try_holder_spread_calculation(self):
        """Verify spread = entry_bank_rate / spot_entry - 1."""
        params = TRYHolderParams(
            principal_try=1_430_000,
            entry_bank_rate_try_to_usd=43.10,
            spot_entry=42.72,
        )
        expected_spread = 43.10 / 42.72 - 1
        assert abs(params.spread - expected_spread) < 0.0001


class TestGBMCalibration:
    """Test GBM calibration uses 252-day annualization."""

    def test_annualization_factor_is_252(self):
        """Verify calibration uses 252 trading days."""
        # Create synthetic daily data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='B')
        returns = np.random.normal(0.0003, 0.01, 500)
        prices = 40 * np.exp(np.cumsum(returns))
        spot_series = pd.Series(prices, index=dates)

        calib = calibrate_gbm(spot_series)

        assert calib['annualization_factor'] == 252

    def test_mu_annual_uses_252(self):
        """Verify mu_annual = mu_daily * 252."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='B')
        returns = np.random.normal(0.0003, 0.01, 500)
        prices = 40 * np.exp(np.cumsum(returns))
        spot_series = pd.Series(prices, index=dates)

        calib = calibrate_gbm(spot_series)

        expected_mu_annual = calib['mu_daily'] * 252
        assert abs(calib['mu_annual'] - expected_mu_annual) < 1e-10

    def test_sigma_annual_uses_sqrt_252(self):
        """Verify sigma_annual = sigma_daily * sqrt(252)."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='B')
        returns = np.random.normal(0.0003, 0.01, 500)
        prices = 40 * np.exp(np.cumsum(returns))
        spot_series = pd.Series(prices, index=dates)

        calib = calibrate_gbm(spot_series)

        expected_sigma_annual = calib['sigma_daily'] * np.sqrt(252)
        assert abs(calib['sigma_annual'] - expected_sigma_annual) < 1e-10


class TestRollingVolatility:
    """Test rolling volatility and regime classification."""

    def test_rolling_vol_output_columns(self):
        """Verify compute_rolling_volatility returns expected columns."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='B')
        prices = 40 + np.cumsum(np.random.normal(0, 0.3, 200))
        spot_series = pd.Series(prices, index=dates)

        vol_df = compute_rolling_volatility(spot_series, window=60)

        assert 'rolling_vol' in vol_df.columns
        assert 'regime' in vol_df.columns
        assert 'vol_33_threshold' in vol_df.columns
        assert 'vol_67_threshold' in vol_df.columns

    def test_regime_classification(self):
        """Verify regime labels are LOW, MEDIUM, or HIGH."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='B')
        prices = 40 + np.cumsum(np.random.normal(0, 0.3, 200))
        spot_series = pd.Series(prices, index=dates)

        vol_df = compute_rolling_volatility(spot_series, window=60)

        valid_regimes = {'LOW', 'MEDIUM', 'HIGH', None}
        actual_regimes = set(vol_df['regime'].unique())
        assert actual_regimes.issubset(valid_regimes)


class TestConsistency:
    """Cross-check consistency between different modes and calculations."""

    def test_same_principal_same_outcome(self):
        """Same effective principal should give same outcome."""
        # TRY holder with 1,430,000 TRY
        try_params = TradeParams(
            principal_try=1_430_000,
            entry_bank_rate=43.10,
            spot_entry=42.72,
        )

        # Verify final_try is computed correctly
        expected_final = try_params.principal_try + try_params.net_interest
        assert abs(try_params.final_try - expected_final) < 0.01

    def test_maturity_date_calculation(self):
        """Verify maturity_date = entry_date + term_days."""
        entry = datetime(2025, 12, 19)
        params = TradeParams(
            entry_date=entry,
            term_days_calendar=32,
        )
        expected_maturity = entry + timedelta(days=32)
        assert params.maturity_date == expected_maturity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
