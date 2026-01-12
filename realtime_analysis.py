"""
realtime_analysis.py
====================
Real-time P&L tracking, trend analysis, and acceleration metrics.

Features:
- Realized P&L from entry date
- USD/TRY trend (slope) analysis
- Acceleration (change in slope)
- Daily cushion calculation
- Forward expected return adjustment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from dashboard_core import TradeParams, CarryTradeModel


@dataclass
class RealTimePnL:
    """Real-time P&L from entry to current."""
    entry_date: datetime
    current_date: datetime
    days_elapsed: int
    days_remaining: int

    # Entry values
    entry_spot: float
    entry_bank_rate: float
    usd0_exec: float

    # Current values
    current_spot: float
    current_bank_rate: float

    # Accrued position
    accrued_interest_gross: float
    accrued_interest_net: float
    current_try_value: float

    # If converting now
    current_usd_value: float
    unrealized_pnl_usd: float
    unrealized_return_pct: float

    # vs T-Bill benchmark
    tbill_value_now: float
    excess_vs_tbill_now: float

    # Mark-to-market
    mtm_pnl_usd: float
    mtm_return_pct: float


@dataclass
class TrendAnalysis:
    """USD/TRY trend and acceleration analysis."""
    # Recent trend (e.g., last 20 days)
    slope_daily: float  # TRY/USD per day
    slope_annual: float  # Annualized
    slope_pct_daily: float  # % per day

    # Acceleration (change in slope)
    accel_daily: float  # Change in slope per day
    accel_regime: str  # "accelerating", "decelerating", "stable"

    # Historical context
    slope_1w: float
    slope_2w: float
    slope_1m: float

    # R-squared of trend fit
    r_squared: float

    # Projected spot at maturity (linear extrapolation)
    projected_spot_maturity: float
    projected_move_pct: float


@dataclass
class DailyCushion:
    """Daily cushion analysis."""
    spot_current: float
    spot_be: float
    cushion_absolute: float  # spot_be - spot_current
    cushion_pct: float  # as % of current spot

    days_remaining: int
    daily_cushion_absolute: float  # cushion / days
    daily_cushion_pct: float  # % per day allowed depreciation

    # Risk assessment
    trend_daily_pct: float  # Current USD/TRY drift
    cushion_vs_trend_ratio: float  # daily_cushion / trend (if <1, trouble)
    days_until_breakeven: float  # At current trend, how many days until BE

    status: str  # "SAFE", "WARNING", "DANGER"


def fetch_entry_and_current(
    spot_series: pd.Series,
    entry_date: str = "2025-12-19"
) -> Dict:
    """
    Fetch spot rates at entry and current.

    Returns dict with entry_date, entry_spot, current_date, current_spot
    """
    # Parse entry date
    entry_dt = pd.Timestamp(entry_date)

    # Handle timezone if present
    if spot_series.index.tz is not None:
        entry_dt = entry_dt.tz_localize(spot_series.index.tz)

    # Find nearest available date to entry
    available_dates = spot_series.index

    # Try exact match first
    if entry_dt in available_dates:
        actual_entry_date = entry_dt
    else:
        # Find nearest
        for delta in range(0, 10):
            for direction in [0, 1, -1]:
                check = entry_dt + pd.Timedelta(days=delta * direction)
                if check in available_dates:
                    actual_entry_date = check
                    break
            else:
                continue
            break
        else:
            # Fallback to nearest
            idx = available_dates.get_indexer([entry_dt], method='nearest')[0]
            actual_entry_date = available_dates[idx]

    entry_spot = spot_series[actual_entry_date]
    current_date = spot_series.index[-1]
    current_spot = spot_series.iloc[-1]

    return {
        'entry_date': actual_entry_date,
        'entry_spot': entry_spot,
        'current_date': current_date,
        'current_spot': current_spot,
        'days_elapsed': (current_date - actual_entry_date).days,
    }


def compute_realtime_pnl(
    params: TradeParams,
    spot_series: pd.Series,
    entry_date: str = "2025-12-19"
) -> RealTimePnL:
    """
    Compute real-time P&L from entry to now.
    """
    # Get spot data
    data = fetch_entry_and_current(spot_series, entry_date)

    entry_spot = data['entry_spot']
    current_spot = data['current_spot']
    days_elapsed = data['days_elapsed']
    days_remaining = params.term_days_calendar - days_elapsed

    # Entry bank rate (derive from spot + spread, or use override)
    entry_bank_rate = entry_spot * (1 + params.spread)

    # Executable USD at entry
    usd0_exec = params.principal_try / entry_bank_rate

    # Current bank rate
    current_bank_rate = current_spot * (1 + params.spread)

    # Accrued interest (pro-rata)
    accrued_gross = params.principal_try * params.deposit_rate_annual * (days_elapsed / 365)
    accrued_stopaj = accrued_gross * params.stopaj_rate
    accrued_net = accrued_gross - accrued_stopaj

    # Current TRY value (principal + accrued)
    current_try_value = params.principal_try + accrued_net

    # If converting now
    current_usd_value = current_try_value / current_bank_rate - params.swift_fee_usd

    # Unrealized P&L
    unrealized_pnl = current_usd_value - usd0_exec
    unrealized_return = (current_usd_value / usd0_exec - 1) * 100

    # T-Bill benchmark (what you'd have if you converted at entry and bought T-Bill)
    tbill_value_now = usd0_exec * (1 + params.usd_rf_rate_annual * days_elapsed / 365)
    excess_vs_tbill = current_usd_value - tbill_value_now

    # Mark-to-market (using current spot but final interest)
    # This shows what you'd get if you held to maturity at current spot
    final_try = params.final_try  # Principal + full net interest at maturity
    mtm_usd = final_try / current_bank_rate - params.swift_fee_usd
    mtm_pnl = mtm_usd - usd0_exec
    mtm_return = (mtm_usd / usd0_exec - 1) * 100

    return RealTimePnL(
        entry_date=data['entry_date'],
        current_date=data['current_date'],
        days_elapsed=days_elapsed,
        days_remaining=max(0, days_remaining),
        entry_spot=entry_spot,
        entry_bank_rate=entry_bank_rate,
        usd0_exec=usd0_exec,
        current_spot=current_spot,
        current_bank_rate=current_bank_rate,
        accrued_interest_gross=accrued_gross,
        accrued_interest_net=accrued_net,
        current_try_value=current_try_value,
        current_usd_value=current_usd_value,
        unrealized_pnl_usd=unrealized_pnl,
        unrealized_return_pct=unrealized_return,
        tbill_value_now=tbill_value_now,
        excess_vs_tbill_now=excess_vs_tbill,
        mtm_pnl_usd=mtm_pnl,
        mtm_return_pct=mtm_return,
    )


def compute_trend_analysis(
    spot_series: pd.Series,
    days_remaining: int,
    current_spot: float
) -> TrendAnalysis:
    """
    Analyze USD/TRY trend and acceleration.

    Trend = slope of USD/TRY over time
    Acceleration = change in slope (second derivative)
    """
    # Get recent data
    recent = spot_series.tail(60)  # Last ~3 months

    # Compute slopes for different windows
    def compute_slope(series: pd.Series) -> Tuple[float, float]:
        """Compute linear regression slope and R-squared."""
        if len(series) < 2:
            return 0.0, 0.0

        x = np.arange(len(series))
        y = series.values

        # Linear regression
        n = len(x)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x ** 2).sum()

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        # R-squared
        y_pred = slope * x + (sum_y - slope * sum_x) / n
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return slope, r2

    # Recent trend (20 trading days ~ 1 month)
    slope_20d, r2_20d = compute_slope(recent.tail(20))

    # 1 week trend
    slope_1w, _ = compute_slope(recent.tail(5))

    # 2 week trend
    slope_2w, _ = compute_slope(recent.tail(10))

    # 1 month trend
    slope_1m, _ = compute_slope(recent.tail(22))

    # Acceleration: change in slope
    # Compare recent slope (last 10 days) vs prior slope (10-20 days ago)
    recent_10 = recent.tail(10)
    prior_10 = recent.iloc[-20:-10] if len(recent) >= 20 else recent.head(10)

    slope_recent, _ = compute_slope(recent_10)
    slope_prior, _ = compute_slope(prior_10)

    accel_daily = slope_recent - slope_prior

    # Determine acceleration regime
    accel_threshold = 0.005  # 0.5 kuruş per day change
    if accel_daily > accel_threshold:
        accel_regime = "ACCELERATING"  # TRY depreciating faster
    elif accel_daily < -accel_threshold:
        accel_regime = "DECELERATING"  # TRY depreciation slowing
    else:
        accel_regime = "STABLE"

    # Annualize slope
    slope_annual = slope_20d * 252
    slope_pct_daily = (slope_20d / current_spot) * 100 if current_spot > 0 else 0

    # Project spot at maturity (linear extrapolation)
    projected_spot = current_spot + slope_20d * days_remaining
    projected_move_pct = (projected_spot / current_spot - 1) * 100

    return TrendAnalysis(
        slope_daily=slope_20d,
        slope_annual=slope_annual,
        slope_pct_daily=slope_pct_daily,
        accel_daily=accel_daily,
        accel_regime=accel_regime,
        slope_1w=slope_1w,
        slope_2w=slope_2w,
        slope_1m=slope_1m,
        r_squared=r2_20d,
        projected_spot_maturity=projected_spot,
        projected_move_pct=projected_move_pct,
    )


def compute_daily_cushion(
    params: TradeParams,
    current_spot: float,
    days_remaining: int,
    trend: TrendAnalysis
) -> DailyCushion:
    """
    Compute daily cushion - how much TRY can depreciate per day before break-even.
    """
    spot_be = params.spot_be

    # Absolute cushion
    cushion_abs = spot_be - current_spot
    cushion_pct = (cushion_abs / current_spot) * 100

    # Daily cushion
    if days_remaining > 0:
        daily_cushion_abs = cushion_abs / days_remaining
        daily_cushion_pct = cushion_pct / days_remaining
    else:
        daily_cushion_abs = cushion_abs
        daily_cushion_pct = cushion_pct

    # Trend comparison
    trend_daily_pct = trend.slope_pct_daily

    # Ratio: if <1, current trend exceeds cushion
    if trend_daily_pct > 0:
        cushion_vs_trend = daily_cushion_pct / trend_daily_pct
    else:
        cushion_vs_trend = float('inf')  # Trend is favorable

    # Days until break-even at current trend
    if trend.slope_daily > 0:
        days_until_be = cushion_abs / trend.slope_daily
    else:
        days_until_be = float('inf')  # Never hits BE if trend is favorable

    # Status assessment
    if cushion_abs <= 0:
        status = "DANGER - ALREADY PAST BREAK-EVEN"
    elif days_until_be < days_remaining:
        status = "DANGER - Trend exceeds cushion"
    elif cushion_vs_trend < 1.5:
        status = "WARNING - Tight cushion vs trend"
    elif cushion_vs_trend < 3:
        status = "CAUTION - Monitor closely"
    else:
        status = "SAFE - Adequate cushion"

    return DailyCushion(
        spot_current=current_spot,
        spot_be=spot_be,
        cushion_absolute=cushion_abs,
        cushion_pct=cushion_pct,
        days_remaining=days_remaining,
        daily_cushion_absolute=daily_cushion_abs,
        daily_cushion_pct=daily_cushion_pct,
        trend_daily_pct=trend_daily_pct,
        cushion_vs_trend_ratio=cushion_vs_trend,
        days_until_breakeven=days_until_be,
        status=status,
    )


def compute_adjusted_expected_return(
    params: TradeParams,
    trend: TrendAnalysis,
    base_mc_results: Dict,
    days_remaining: int
) -> Dict:
    """
    Adjust expected return based on current trend/acceleration.

    Method: Blend historical calibration with recent trend momentum.
    """
    # Base expected return from MC
    base_expected = base_mc_results.get('mean_excess', 0)
    base_prob_under = base_mc_results.get('prob_underperform_tbill', 50)

    # Trend-adjusted drift
    # If recent trend is stronger than historical mean, adjust
    historical_mu = base_mc_results.get('mu_annual', 0.18)  # ~18% default
    recent_mu = trend.slope_annual / params.spot_entry if params.spot_entry > 0 else 0

    # Blend: weight recent trend more if R-squared is high
    r2 = trend.r_squared
    blended_mu = historical_mu * (1 - r2 * 0.5) + recent_mu * (r2 * 0.5)

    # Acceleration adjustment
    # If accelerating, expect worse outcomes
    accel_factor = 1.0
    if trend.accel_regime == "ACCELERATING":
        accel_factor = 1.2  # 20% worse expected outcome
    elif trend.accel_regime == "DECELERATING":
        accel_factor = 0.8  # 20% better

    # Adjusted expected return
    # Project spot at maturity using blended drift
    T = days_remaining / 365
    projected_spot_drift = params.spot_entry * np.exp(blended_mu * T)

    # Compute outcome at projected spot
    model = CarryTradeModel(params)
    outcome_projected = model.compute_outcome(projected_spot_drift)

    # Apply acceleration factor to the excess return
    adjusted_excess = outcome_projected.excess_ret_vs_tbill / accel_factor

    # Adjusted probability (heuristic based on cushion)
    cushion_factor = (params.spot_be - params.spot_entry) / params.spot_entry
    trend_factor = trend.projected_move_pct / 100

    if trend_factor > cushion_factor:
        # Trend projects past break-even
        adjusted_prob_under = min(95, base_prob_under * 1.3)
    else:
        adjusted_prob_under = base_prob_under

    return {
        'base_expected_excess': base_expected,
        'adjusted_expected_excess': adjusted_excess,
        'adjustment_factor': accel_factor,
        'blended_mu_annual': blended_mu * 100,
        'historical_mu_annual': historical_mu * 100,
        'recent_mu_annual': recent_mu * 100,
        'base_prob_underperform': base_prob_under,
        'adjusted_prob_underperform': adjusted_prob_under,
        'projected_spot_at_maturity': projected_spot_drift,
        'trend_contribution': trend.projected_move_pct,
    }


def generate_realtime_report(
    pnl: RealTimePnL,
    trend: TrendAnalysis,
    cushion: DailyCushion,
    adjusted: Dict
) -> str:
    """Generate comprehensive real-time analysis report."""

    lines = [
        "=" * 70,
        "REAL-TIME CARRY TRADE ANALYSIS",
        "=" * 70,
        "",
        "POSITION STATUS",
        "-" * 70,
        f"  Entry Date:           {pnl.entry_date.strftime('%Y-%m-%d')}",
        f"  Current Date:         {pnl.current_date.strftime('%Y-%m-%d')}",
        f"  Days Elapsed:         {pnl.days_elapsed}",
        f"  Days Remaining:       {pnl.days_remaining}",
        "",
        "ENTRY VALUES",
        "-" * 70,
        f"  Entry Spot:           {pnl.entry_spot:.4f}",
        f"  Entry Bank Rate:      {pnl.entry_bank_rate:.4f}",
        f"  USD Invested:         ${pnl.usd0_exec:,.2f}",
        "",
        "CURRENT VALUES",
        "-" * 70,
        f"  Current Spot:         {pnl.current_spot:.4f}",
        f"  Current Bank Rate:    {pnl.current_bank_rate:.4f}",
        f"  Spot Move:            {(pnl.current_spot/pnl.entry_spot - 1)*100:+.2f}%",
        "",
        "ACCRUED POSITION",
        "-" * 70,
        f"  Accrued Interest:     {pnl.accrued_interest_net:,.2f} TRY (net of stopaj)",
        f"  Current TRY Value:    {pnl.current_try_value:,.2f} TRY",
        "",
        "UNREALIZED P&L (if converting now)",
        "-" * 70,
        f"  Current USD Value:    ${pnl.current_usd_value:,.2f}",
        f"  Unrealized P&L:       ${pnl.unrealized_pnl_usd:+,.2f}",
        f"  Unrealized Return:    {pnl.unrealized_return_pct:+.2f}%",
        "",
        "vs T-BILL BENCHMARK",
        "-" * 70,
        f"  T-Bill Value Now:     ${pnl.tbill_value_now:,.2f}",
        f"  Excess vs T-Bill:     ${pnl.excess_vs_tbill_now:+,.2f}",
        "",
        "MARK-TO-MARKET (hold to maturity at current spot)",
        "-" * 70,
        f"  MTM P&L:              ${pnl.mtm_pnl_usd:+,.2f}",
        f"  MTM Return:           {pnl.mtm_return_pct:+.2f}%",
        "",
        "=" * 70,
        "TREND ANALYSIS",
        "=" * 70,
        "",
        f"  1-Week Slope:         {trend.slope_1w:.4f} TRY/day",
        f"  2-Week Slope:         {trend.slope_2w:.4f} TRY/day",
        f"  1-Month Slope:        {trend.slope_1m:.4f} TRY/day",
        f"  20-Day Slope:         {trend.slope_daily:.4f} TRY/day ({trend.slope_pct_daily:.3f}%/day)",
        f"  Annualized Drift:     {trend.slope_annual:.2f} TRY/year ({trend.slope_annual/pnl.current_spot*100:.1f}%)",
        f"  Trend R-squared:      {trend.r_squared:.3f}",
        "",
        "ACCELERATION",
        "-" * 70,
        f"  Daily Acceleration:   {trend.accel_daily:.4f} TRY/day^2",
        f"  Regime:               {trend.accel_regime}",
        "",
        "PROJECTION (linear extrapolation)",
        "-" * 70,
        f"  Projected Spot:       {trend.projected_spot_maturity:.4f}",
        f"  Projected Move:       {trend.projected_move_pct:+.2f}%",
        "",
        "=" * 70,
        "DAILY CUSHION ANALYSIS",
        "=" * 70,
        "",
        f"  Current Spot:         {cushion.spot_current:.4f}",
        f"  Break-even Spot:      {cushion.spot_be:.4f}",
        f"  Cushion (absolute):   {cushion.cushion_absolute:.4f} TRY",
        f"  Cushion (%):          {cushion.cushion_pct:.2f}%",
        "",
        f"  Days Remaining:       {cushion.days_remaining}",
        f"  Daily Cushion:        {cushion.daily_cushion_absolute:.4f} TRY/day",
        f"  Daily Cushion (%):    {cushion.daily_cushion_pct:.3f}%/day",
        "",
        f"  Current Trend:        {cushion.trend_daily_pct:.3f}%/day",
        f"  Cushion/Trend Ratio:  {cushion.cushion_vs_trend_ratio:.2f}x",
        f"  Days Until BE:        {cushion.days_until_breakeven:.1f} days (at current trend)",
        "",
        f"  STATUS:               {cushion.status}",
        "",
        "=" * 70,
        "ADJUSTED EXPECTED RETURN",
        "=" * 70,
        "",
        f"  Historical Drift:     {adjusted['historical_mu_annual']:.1f}% (annualized)",
        f"  Recent Drift:         {adjusted['recent_mu_annual']:.1f}% (annualized)",
        f"  Blended Drift:        {adjusted['blended_mu_annual']:.1f}% (annualized)",
        "",
        f"  Base Expected Excess: {adjusted['base_expected_excess']:+.2f}%",
        f"  Adjustment Factor:    {adjusted['adjustment_factor']:.2f}x",
        f"  Adjusted Expected:    {adjusted['adjusted_expected_excess']:+.2f}%",
        "",
        f"  Base P(Underperform): {adjusted['base_prob_underperform']:.1f}%",
        f"  Adj P(Underperform):  {adjusted['adjusted_prob_underperform']:.1f}%",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample data
    from data import fetch_usdtry_data, get_spot_series
    from dashboard_core import calibrate_gbm

    print("Fetching data...")
    df = fetch_usdtry_data()
    spot_series = get_spot_series(df)

    # Create params
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

    # Compute real-time P&L
    pnl = compute_realtime_pnl(params, spot_series, entry_date="2025-12-19")

    # Compute trend
    trend = compute_trend_analysis(spot_series, pnl.days_remaining, pnl.current_spot)

    # Compute daily cushion
    cushion = compute_daily_cushion(params, pnl.current_spot, pnl.days_remaining, trend)

    # Get base MC results for adjustment
    calib = calibrate_gbm(spot_series, regime_days=365)
    model = CarryTradeModel(params)
    mc_results = model.monte_carlo(calib['mu_annual'], calib['sigma_annual'])

    # Compute adjusted expected return
    adjusted = compute_adjusted_expected_return(params, trend, mc_results, pnl.days_remaining)

    # Generate report
    report = generate_realtime_report(pnl, trend, cushion, adjusted)
    print(report)
