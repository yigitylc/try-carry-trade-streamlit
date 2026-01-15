"""
utils.py
========
Utility functions for formatting, validation, and consistency checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import io
import csv


def format_currency(value: float, currency: str = 'USD', decimals: int = 2) -> str:
    """Format value as currency string."""
    if currency == 'USD':
        return f"${value:,.{decimals}f}"
    elif currency == 'TRY':
        return f"{value:,.{decimals}f} TL"
    else:
        return f"{value:,.{decimals}f} {currency}"


def format_percent(value: float, decimals: int = 2, with_sign: bool = True) -> str:
    """Format value as percentage string."""
    if with_sign:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """Format number with thousands separator."""
    return f"{value:,.{decimals}f}"


def validate_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate trade parameters.

    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []

    # Required positive values
    positive_fields = [
        ('principal_try', 'Principal (TRY)'),
        ('deposit_rate_annual', 'Deposit Rate'),
        ('term_days_calendar', 'Term Days'),
        ('entry_bank_rate', 'Entry Bank Rate'),
        ('spot_entry', 'Spot Entry'),
    ]

    for field, name in positive_fields:
        val = params.get(field)
        if val is None:
            errors.append(f"{name} is required")
        elif val <= 0:
            errors.append(f"{name} must be positive")

    # Rate validations
    if params.get('deposit_rate_annual', 0) > 1:
        errors.append("Deposit rate should be decimal (e.g., 0.395 for 39.5%)")

    if params.get('stopaj_rate', 0) > 1:
        errors.append("Stopaj rate should be decimal (e.g., 0.175 for 17.5%)")

    if params.get('usd_rf_rate_annual', 0) > 1:
        errors.append("US RF rate should be decimal (e.g., 0.035 for 3.5%)")

    # Logical validations
    if params.get('entry_bank_rate', 0) < params.get('spot_entry', 0):
        errors.append("Bank rate should be >= spot (bank sells TRY at premium)")

    if params.get('term_days_calendar', 0) > 365:
        errors.append("Term > 365 days is unusual for this analysis")

    if errors:
        return False, "; ".join(errors)

    return True, ""


def validate_spot_series(spot_series: pd.Series, min_points: int = 60) -> Tuple[bool, str]:
    """
    Validate spot price series.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if spot_series is None or len(spot_series) == 0:
        return False, "Spot series is empty"

    if len(spot_series) < min_points:
        return False, f"Need at least {min_points} data points, got {len(spot_series)}"

    if spot_series.isna().any():
        na_count = spot_series.isna().sum()
        return False, f"Series contains {na_count} missing values"

    if (spot_series <= 0).any():
        return False, "Series contains non-positive values"

    return True, ""


def cross_check_interest(
    principal: float,
    rate_annual: float,
    days: int,
    stopaj_rate: float,
    expected_gross: float = None,
    expected_net: float = None,
    expected_final: float = None,
    tolerance: float = 1.0
) -> Dict[str, Any]:
    """
    Cross-check interest calculations.

    Returns dict with calculated values and match status.
    """
    calc_gross = principal * rate_annual * (days / 365)
    calc_stopaj = calc_gross * stopaj_rate
    calc_net = calc_gross - calc_stopaj
    calc_final = principal + calc_net

    result = {
        'calculated': {
            'gross_interest': calc_gross,
            'stopaj': calc_stopaj,
            'net_interest': calc_net,
            'final_try': calc_final,
        },
        'matches': {}
    }

    if expected_gross is not None:
        result['matches']['gross'] = abs(calc_gross - expected_gross) < tolerance
        result['expected_gross'] = expected_gross

    if expected_net is not None:
        result['matches']['net'] = abs(calc_net - expected_net) < tolerance
        result['expected_net'] = expected_net

    if expected_final is not None:
        result['matches']['final'] = abs(calc_final - expected_final) < tolerance
        result['expected_final'] = expected_final

    result['all_match'] = all(result['matches'].values()) if result['matches'] else True

    return result


def verify_breakeven_logic(
    final_try: float,
    usd_rf_end: float,
    swift_fee: float,
    spread: float,
    spot_entry: float
) -> Dict[str, Any]:
    """
    Verify break-even calculation and direction.

    Returns dict with break-even values and direction verification.
    """
    # Break-even: final_try / bank_rate_be - swift = usd_rf_end
    bank_rate_be = final_try / (usd_rf_end + swift_fee)
    spot_be = bank_rate_be / (1 + spread)
    be_move_pct = (spot_be / spot_entry - 1) * 100

    # Test points
    spot_worse = spot_be * 1.02
    spot_better = spot_be * 0.98

    bank_worse = spot_worse * (1 + spread)
    bank_better = spot_better * (1 + spread)

    usd_worse = final_try / bank_worse - swift_fee
    usd_better = final_try / bank_better - swift_fee

    excess_worse = (usd_worse / usd_rf_end - 1) * 100
    excess_better = (usd_better / usd_rf_end - 1) * 100

    return {
        'bank_rate_be': bank_rate_be,
        'spot_be': spot_be,
        'be_move_pct': be_move_pct,
        'direction_check': {
            'spot_worse_than_be': spot_worse,
            'excess_at_worse': excess_worse,
            'worse_underperforms': excess_worse < 0,  # Should be True
            'spot_better_than_be': spot_better,
            'excess_at_better': excess_better,
            'better_outperforms': excess_better > 0,  # Should be True
        },
        'logic_correct': excess_worse < 0 and excess_better > 0,
    }


def generate_scenario_table_csv(scenarios_df: pd.DataFrame) -> str:
    """
    Generate CSV string from scenario DataFrame.
    """
    output = io.StringIO()
    scenarios_df.to_csv(output, index=False, float_format='%.4f')
    return output.getvalue()


def generate_metrics_csv(params: Dict, mc_results: Dict, backtest: Dict = None) -> str:
    """
    Generate CSV of key metrics.
    """
    rows = [
        ['Category', 'Metric', 'Value'],
        ['Trade', 'Principal TRY', params.get('principal_try', '')],
        ['Trade', 'USD Baseline', params.get('usd0_baseline', '')],
        ['Trade', 'T-Bill End Value', params.get('usd_rf_end', '')],
        ['Trade', 'Break-even Spot', params.get('spot_be', '')],
        ['Trade', 'Break-even Move %', params.get('be_move_pct', '')],
        ['Monte Carlo', 'Mean Excess vs T-Bill %', mc_results.get('mean_excess', '')],
        ['Monte Carlo', 'Median Excess %', mc_results.get('median_excess', '')],
        ['Monte Carlo', 'Std Excess %', mc_results.get('std_excess', '')],
        ['Monte Carlo', 'P(Underperform T-Bill) %', mc_results.get('prob_underperform_tbill', '')],
        ['Monte Carlo', 'VaR 95% Excess %', mc_results.get('var_95_excess', '')],
        ['Monte Carlo', 'CVaR 95% Excess %', mc_results.get('cvar_95_excess', '')],
        ['Monte Carlo', 'Sharpe Ratio', mc_results.get('sharpe', '')],
    ]

    if backtest and 'error' not in backtest:
        rows.extend([
            ['Backtest', 'N Windows', backtest.get('n_windows', '')],
            ['Backtest', 'Mean Excess %', backtest.get('mean_excess', '')],
            ['Backtest', 'Win Rate vs T-Bill %', backtest.get('win_rate_vs_tbill', '')],
        ])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()


def format_regime_name(days: Optional[int]) -> str:
    """Format regime days as readable name."""
    if days == 365:
        return "1 Year"
    elif days == 730:
        return "2 Years"
    elif days == 1095:
        return "3 Years"
    elif days == 1460:
        return "4 Years"
    elif days == 1825:
        return "5 Years"
    else:
        return f"{days} Days"


def get_regime_days(name: str) -> Optional[int]:
    """Convert regime name to days."""
    mapping = {
        '1 Year': 365,
        '1Y': 365,
        '2 Years': 730,
        '2Y': 730,
        '3 Years': 1095,
        '3Y': 1095,
        '4 Years': 1460,
        '4Y': 1460,
        '5 Years': 1825,
        '5Y': 1825,
    }
    return mapping.get(name)


def compute_trade_economics_summary(params: Dict) -> str:
    """
    Generate text summary of trade economics.
    """
    lines = [
        "=" * 60,
        "TRADE ECONOMICS SUMMARY",
        "=" * 60,
        "",
        "ENTRY",
        f"  Mode:               {params['mode']}",
        f"  Principal (TRY):     {format_currency(params['principal_try'], 'TRY', 0)}",
        f"  Principal (USD):     {format_currency(params['principal_usd'])}",
        f"  Entry Spot:          {params['spot_entry']:.4f}",
        f"  Entry Bank Rate:     {params['entry_bank_rate']:.4f}",
        f"  Entry Spread:        {params['entry_spread']*100:.2f}%",
        f"  USD Baseline:        {format_currency(params['usd0_baseline'])}",
        "",
        "DEPOSIT",
        f"  Rate (annual):       {params['deposit_rate_annual']*100:.2f}%",
        f"  Term:                {params['term_days_calendar']} calendar days",
        f"  Stopaj:              {params['stopaj_rate']*100:.2f}%",
        f"  Gross Interest:      {format_currency(params['gross_interest'], 'TRY')}",
        f"  Net Interest:        {format_currency(params['net_interest'], 'TRY')}",
        f"  Final TRY:           {format_currency(params['final_try'], 'TRY')}",
        "",
        "BENCHMARK (T-Bill)",
        f"  US RF Rate:          {params['usd_rf_rate_annual']*100:.2f}%",
        f"  T-Bill End Value:    {format_currency(params['usd_rf_end'])}",
        f"  T-Bill Return:       {format_percent(params['rf_return_period']*100)}",
        "",
        "BREAK-EVEN",
        f"  Break-even Spot:     {params['spot_be']:.4f}",
        f"  Required Move:       {format_percent(params['be_move_pct'])}",
        f"  Interpretation:      If USD/TRY rises more than {params['be_move_pct']:.2f}%,",
        f"                       carry trade underperforms T-Bill.",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def print_consistency_report(checks: Dict) -> str:
    """
    Format consistency check results as readable report.
    """
    lines = [
        "=" * 60,
        "INTERNAL CONSISTENCY CHECK",
        "=" * 60,
        "",
        "INPUTS:",
    ]

    for k, v in checks['inputs'].items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:,.4f}")
        else:
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("COMPUTED VALUES:")

    for k, v in checks['computed'].items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:,.4f}")
        else:
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("VERIFICATION:")

    for k, v in checks['verification'].items():
        if isinstance(v, bool):
            status = "PASS" if v else "FAIL"
            lines.append(f"  {k}: {status}")
        elif isinstance(v, float):
            lines.append(f"  {k}: {v:,.4f}")
        else:
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("=" * 60)

    if checks['verification'].get('ALL_CHECKS_PASS', False):
        lines.append("ALL CONSISTENCY CHECKS PASSED")
    else:
        lines.append("SOME CHECKS FAILED - REVIEW MODEL")

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test formatting functions
    print(format_currency(33178.654, 'USD'))
    print(format_currency(1430000, 'TRY', 0))
    print(format_percent(2.57))
    print(format_percent(-1.23))
    print(format_number(43.1070))
