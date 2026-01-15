"""
dashboard_core.py
=================
Core computation module for TRY Carry Trade Analysis.

CRITICAL MODEL: Opportunity-Cost Baseline
-----------------------------------------
Baseline = What USD could be held risk-free over the same horizon.

Mode-specific baselines:
- TRY-holder: USD baseline = TRY0 / entry_bank_rate_try_to_usd (optionally minus swift)
- USD-holder: USD baseline = USD0 (already in USD)

Variables:
- usd0_baseline: USD baseline at entry
- usd_rf_end: T-bill end value = usd0_baseline * (1 + rf * T)
- usd_end: Carry trade outcome = final_try / bank_rate_end - swift_fee
- bank_rate_be: Break-even bank rate where carry = T-bill
- pnl_vs_baseline: usd_end - usd0_baseline
- pnl_vs_tbill: usd_end - usd_rf_end (excess P/L)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


@dataclass
class TryHolderParams:
    """Inputs for TRY-holder mode."""
    principal_try: float
    entry_bank_rate_try_to_usd: float
    spot_entry: float
    start_date: datetime
    term_days_calendar: int
    deposit_rate_annual: float
    stopaj_rate: float
    swift_fee_usd: float
    usd_rf_rate_annual: float
    exit_spread: float
    use_compound_interest: bool = False
    include_swift_in_baseline: bool = False
    bank_rate_end_override: Optional[float] = None


@dataclass
class UsdHolderParams:
    """Inputs for USD-holder mode."""
    principal_usd: float
    entry_bank_rate_usd_to_try: float
    spot_entry: float
    start_date: datetime
    term_days_calendar: int
    deposit_rate_annual: float
    stopaj_rate: float
    swift_fee_usd: float
    usd_rf_rate_annual: float
    exit_spread: float
    use_compound_interest: bool = False
    bank_rate_end_override: Optional[float] = None


@dataclass
class TradeMetrics:
    """Computed metrics used across the dashboard."""
    mode: str
    principal_try: float
    principal_usd: float
    entry_bank_rate: float
    entry_spread: float
    spot_entry: float
    start_date: datetime
    maturity_date: datetime
    term_days_calendar: int
    term_years: float
    deposit_rate_annual: float
    stopaj_rate: float
    net_deposit_rate_annual: float
    gross_interest: float
    net_interest: float
    final_try: float
    swift_fee_usd: float
    usd_rf_rate_annual: float
    usd0_baseline: float
    usd_rf_end: float
    rf_return_period: float
    use_compound_interest: bool
    exit_spread: float
    bank_rate_end_override: Optional[float]
    bank_rate_be: float
    spot_be: float
    be_move_pct: float
    include_swift_in_baseline: bool = False


@dataclass
class CarryOutcome:
    """Outcome of carry trade at a given maturity spot."""
    spot_end: float
    bank_rate_end: float
    usd_end: float
    pnl_vs_baseline: float
    pnl_vs_tbill: float
    ret_vs_baseline: float
    excess_ret_vs_tbill: float
    beats_tbill: bool


def _compute_interest(principal_try: float, deposit_rate_annual: float, stopaj_rate: float, term_years: float) -> Tuple[float, float, float, float]:
    gross_interest = principal_try * deposit_rate_annual * term_years
    stopaj_amount = gross_interest * stopaj_rate
    net_interest = gross_interest - stopaj_amount
    final_try = principal_try + net_interest
    return gross_interest, stopaj_amount, net_interest, final_try


def compute_try_holder_metrics(params: TryHolderParams) -> TradeMetrics:
    """Compute metrics for TRY-holder mode."""
    term_years = params.term_days_calendar / CALENDAR_DAYS_PER_YEAR
    gross_interest, _, net_interest, final_try = _compute_interest(
        params.principal_try,
        params.deposit_rate_annual,
        params.stopaj_rate,
        term_years,
    )

    usd0_baseline = params.principal_try / params.entry_bank_rate_try_to_usd
    if params.include_swift_in_baseline:
        usd0_baseline -= params.swift_fee_usd

    if params.use_compound_interest:
        usd_rf_end = usd0_baseline * (1 + params.usd_rf_rate_annual) ** term_years
    else:
        usd_rf_end = usd0_baseline * (1 + params.usd_rf_rate_annual * term_years)

    rf_return_period = usd_rf_end / usd0_baseline - 1

    bank_rate_be = final_try / (usd_rf_end + params.swift_fee_usd)
    spot_be = bank_rate_be / (1 + params.exit_spread)
    be_move_pct = (spot_be / params.spot_entry - 1) * 100

    entry_spread = params.entry_bank_rate_try_to_usd / params.spot_entry - 1

    maturity_date = params.start_date + timedelta(days=params.term_days_calendar)

    return TradeMetrics(
        mode="try_holder",
        principal_try=params.principal_try,
        principal_usd=usd0_baseline,
        entry_bank_rate=params.entry_bank_rate_try_to_usd,
        entry_spread=entry_spread,
        spot_entry=params.spot_entry,
        start_date=params.start_date,
        maturity_date=maturity_date,
        term_days_calendar=params.term_days_calendar,
        term_years=term_years,
        deposit_rate_annual=params.deposit_rate_annual,
        stopaj_rate=params.stopaj_rate,
        net_deposit_rate_annual=params.deposit_rate_annual * (1 - params.stopaj_rate),
        gross_interest=gross_interest,
        net_interest=net_interest,
        final_try=final_try,
        swift_fee_usd=params.swift_fee_usd,
        usd_rf_rate_annual=params.usd_rf_rate_annual,
        usd0_baseline=usd0_baseline,
        usd_rf_end=usd_rf_end,
        rf_return_period=rf_return_period,
        use_compound_interest=params.use_compound_interest,
        exit_spread=params.exit_spread,
        bank_rate_end_override=params.bank_rate_end_override,
        bank_rate_be=bank_rate_be,
        spot_be=spot_be,
        be_move_pct=be_move_pct,
        include_swift_in_baseline=params.include_swift_in_baseline,
    )


def compute_usd_holder_metrics(params: UsdHolderParams) -> TradeMetrics:
    """Compute metrics for USD-holder mode."""
    term_years = params.term_days_calendar / CALENDAR_DAYS_PER_YEAR
    principal_try = params.principal_usd * params.entry_bank_rate_usd_to_try
    gross_interest, _, net_interest, final_try = _compute_interest(
        principal_try,
        params.deposit_rate_annual,
        params.stopaj_rate,
        term_years,
    )

    usd0_baseline = params.principal_usd

    if params.use_compound_interest:
        usd_rf_end = usd0_baseline * (1 + params.usd_rf_rate_annual) ** term_years
    else:
        usd_rf_end = usd0_baseline * (1 + params.usd_rf_rate_annual * term_years)

    rf_return_period = usd_rf_end / usd0_baseline - 1

    bank_rate_be = final_try / (usd_rf_end + params.swift_fee_usd)
    spot_be = bank_rate_be / (1 + params.exit_spread)
    be_move_pct = (spot_be / params.spot_entry - 1) * 100

    entry_spread = params.entry_bank_rate_usd_to_try / params.spot_entry - 1

    maturity_date = params.start_date + timedelta(days=params.term_days_calendar)

    return TradeMetrics(
        mode="usd_holder",
        principal_try=principal_try,
        principal_usd=params.principal_usd,
        entry_bank_rate=params.entry_bank_rate_usd_to_try,
        entry_spread=entry_spread,
        spot_entry=params.spot_entry,
        start_date=params.start_date,
        maturity_date=maturity_date,
        term_days_calendar=params.term_days_calendar,
        term_years=term_years,
        deposit_rate_annual=params.deposit_rate_annual,
        stopaj_rate=params.stopaj_rate,
        net_deposit_rate_annual=params.deposit_rate_annual * (1 - params.stopaj_rate),
        gross_interest=gross_interest,
        net_interest=net_interest,
        final_try=final_try,
        swift_fee_usd=params.swift_fee_usd,
        usd_rf_rate_annual=params.usd_rf_rate_annual,
        usd0_baseline=usd0_baseline,
        usd_rf_end=usd_rf_end,
        rf_return_period=rf_return_period,
        use_compound_interest=params.use_compound_interest,
        exit_spread=params.exit_spread,
        bank_rate_end_override=params.bank_rate_end_override,
        bank_rate_be=bank_rate_be,
        spot_be=spot_be,
        be_move_pct=be_move_pct,
        include_swift_in_baseline=False,
    )


class CarryTradeModel:
    """Models TRY carry trade outcomes for either mode."""

    def __init__(self, metrics: TradeMetrics):
        self.metrics = metrics

    def compute_outcome(self, spot_end: float, bank_rate_end: Optional[float] = None) -> CarryOutcome:
        """Compute carry trade outcome for given maturity spot."""
        m = self.metrics

        if bank_rate_end is not None:
            br_end = bank_rate_end
        elif m.bank_rate_end_override is not None:
            br_end = m.bank_rate_end_override
        else:
            br_end = spot_end * (1 + m.exit_spread)

        usd_end = m.final_try / br_end - m.swift_fee_usd

        pnl_vs_baseline = usd_end - m.usd0_baseline
        pnl_vs_tbill = usd_end - m.usd_rf_end

        ret_vs_baseline = (usd_end / m.usd0_baseline - 1) * 100
        excess_ret_vs_tbill = (usd_end / m.usd_rf_end - 1) * 100

        return CarryOutcome(
            spot_end=spot_end,
            bank_rate_end=br_end,
            usd_end=usd_end,
            pnl_vs_baseline=pnl_vs_baseline,
            pnl_vs_tbill=pnl_vs_tbill,
            ret_vs_baseline=ret_vs_baseline,
            excess_ret_vs_tbill=excess_ret_vs_tbill,
            beats_tbill=pnl_vs_tbill > 0,
        )

    def scenario_analysis(self, spot_moves_pct: Optional[List[float]] = None) -> pd.DataFrame:
        """Run scenario analysis for various spot movements."""
        if spot_moves_pct is None:
            be = self.metrics.be_move_pct
            spot_moves_pct = [x / 2 for x in range(-20, 21)]
            spot_moves_pct.append(round(be, 2))
            spot_moves_pct = sorted(set(spot_moves_pct))

        results = []
        for move in spot_moves_pct:
            spot_end = self.metrics.spot_entry * (1 + move / 100)
            outcome = self.compute_outcome(spot_end)

            results.append({
                'spot_move_pct': move,
                'spot_end': spot_end,
                'bank_rate_end': outcome.bank_rate_end,
                'usd_end': outcome.usd_end,
                'pnl_vs_baseline': outcome.pnl_vs_baseline,
                'pnl_vs_tbill': outcome.pnl_vs_tbill,
                'ret_vs_baseline': outcome.ret_vs_baseline,
                'excess_ret_vs_tbill': outcome.excess_ret_vs_tbill,
                'beats_tbill': outcome.beats_tbill,
                'is_breakeven': abs(move - self.metrics.be_move_pct) < 0.1,
            })

        return pd.DataFrame(results)

    def monte_carlo(
        self,
        mu_annual: float,
        sigma_annual: float,
        n_sims: int = 50000,
        seed: int = 42,
    ) -> Dict:
        """Monte Carlo simulation using GBM with trading-day horizon."""
        m = self.metrics

        trading_days = max(1, int(round(m.term_days_calendar * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR)))
        T = trading_days / TRADING_DAYS_PER_YEAR

        np.random.seed(seed)
        Z = np.random.standard_normal(n_sims)

        drift = (mu_annual - 0.5 * sigma_annual ** 2) * T
        diffusion = sigma_annual * np.sqrt(T) * Z

        spot_ends = m.spot_entry * np.exp(drift + diffusion)

        ret_vs_baseline = np.zeros(n_sims)
        excess_ret = np.zeros(n_sims)
        usd_ends = np.zeros(n_sims)

        for i, spot_end in enumerate(spot_ends):
            outcome = self.compute_outcome(spot_end)
            ret_vs_baseline[i] = outcome.ret_vs_baseline
            excess_ret[i] = outcome.excess_ret_vs_tbill
            usd_ends[i] = outcome.usd_end

        prob_underperform_tbill = (excess_ret < 0).mean() * 100
        prob_loss_vs_baseline = (ret_vs_baseline < 0).mean() * 100

        var_95_excess = np.percentile(excess_ret, 5)
        var_99_excess = np.percentile(excess_ret, 1)
        cvar_95_excess = excess_ret[excess_ret <= var_95_excess].mean()
        cvar_99_excess = excess_ret[excess_ret <= var_99_excess].mean()

        var_95_ret = np.percentile(ret_vs_baseline, 5)
        var_99_ret = np.percentile(ret_vs_baseline, 1)
        cvar_95_ret = ret_vs_baseline[ret_vs_baseline <= var_95_ret].mean()
        cvar_99_ret = ret_vs_baseline[ret_vs_baseline <= var_99_ret].mean()

        sharpe = excess_ret.mean() / excess_ret.std() if excess_ret.std() > 0 else 0

        return {
            'spot_ends': spot_ends,
            'usd_ends': usd_ends,
            'ret_vs_baseline': ret_vs_baseline,
            'excess_ret': excess_ret,
            'n_sims': n_sims,
            'mu_annual': mu_annual,
            'sigma_annual': sigma_annual,
            'mean_ret': ret_vs_baseline.mean(),
            'median_ret': np.median(ret_vs_baseline),
            'std_ret': ret_vs_baseline.std(),
            'p5_ret': np.percentile(ret_vs_baseline, 5),
            'p95_ret': np.percentile(ret_vs_baseline, 95),
            'var_95_ret': var_95_ret,
            'var_99_ret': var_99_ret,
            'cvar_95_ret': cvar_95_ret,
            'cvar_99_ret': cvar_99_ret,
            'prob_loss_vs_baseline': prob_loss_vs_baseline,
            'mean_excess': excess_ret.mean(),
            'median_excess': np.median(excess_ret),
            'std_excess': excess_ret.std(),
            'p5_excess': np.percentile(excess_ret, 5),
            'p95_excess': np.percentile(excess_ret, 95),
            'var_95_excess': var_95_excess,
            'var_99_excess': var_99_excess,
            'cvar_95_excess': cvar_95_excess,
            'cvar_99_excess': cvar_99_excess,
            'prob_underperform_tbill': prob_underperform_tbill,
            'sharpe': sharpe,
            'mean_usd_end': usd_ends.mean(),
            'mean_pnl_vs_baseline': usd_ends.mean() - m.usd0_baseline,
            'mean_pnl_vs_tbill': usd_ends.mean() - m.usd_rf_end,
        }

    def historical_backtest(
        self,
        spot_series: pd.Series,
        regime_days: Optional[int] = None,
    ) -> Dict:
        """Historical backtest using rolling calendar-day windows."""
        m = self.metrics

        if regime_days is not None:
            cutoff = spot_series.index[-1] - pd.Timedelta(days=regime_days)
            spot_series = spot_series[spot_series.index >= cutoff]

        if len(spot_series) < 60:
            return {'error': 'Insufficient data', 'n_windows': 0}

        results = []
        dates = spot_series.index

        for start_date in dates:
            target_end = start_date + pd.Timedelta(days=m.term_days_calendar)
            future_dates = dates[dates >= target_end]
            if len(future_dates) == 0:
                continue

            end_date = future_dates[0]

            spot_start = spot_series[start_date]
            spot_end = spot_series[end_date]

            elapsed_days = (end_date - start_date).days
            term_years = elapsed_days / CALENDAR_DAYS_PER_YEAR

            bank_rate_entry_t = spot_start * (1 + m.entry_spread)

            if m.mode == "try_holder":
                principal_try_t = m.principal_try
                usd0_baseline_t = principal_try_t / bank_rate_entry_t
                if m.include_swift_in_baseline:
                    usd0_baseline_t -= m.swift_fee_usd
            else:
                principal_try_t = m.principal_usd * bank_rate_entry_t
                usd0_baseline_t = m.principal_usd

            gross_interest_t, _, net_interest_t, final_try_t = _compute_interest(
                principal_try_t,
                m.deposit_rate_annual,
                m.stopaj_rate,
                term_years,
            )

            if m.use_compound_interest:
                usd_rf_end_t = usd0_baseline_t * (1 + m.usd_rf_rate_annual) ** term_years
            else:
                usd_rf_end_t = usd0_baseline_t * (1 + m.usd_rf_rate_annual * term_years)

            bank_rate_end_t = spot_end * (1 + m.exit_spread)
            usd_end_t = final_try_t / bank_rate_end_t - m.swift_fee_usd

            pnl_vs_baseline = usd_end_t - usd0_baseline_t
            pnl_vs_tbill = usd_end_t - usd_rf_end_t
            ret_vs_baseline = (usd_end_t / usd0_baseline_t - 1) * 100
            excess_ret = (usd_end_t / usd_rf_end_t - 1) * 100

            results.append({
                'start_date': start_date,
                'end_date': end_date,
                'spot_start': spot_start,
                'spot_end': spot_end,
                'spot_move_pct': (spot_end / spot_start - 1) * 100,
                'usd0_baseline_t': usd0_baseline_t,
                'usd_rf_end_t': usd_rf_end_t,
                'usd_end_t': usd_end_t,
                'pnl_vs_baseline': pnl_vs_baseline,
                'pnl_vs_tbill': pnl_vs_tbill,
                'ret_vs_baseline': ret_vs_baseline,
                'excess_ret': excess_ret,
                'beats_tbill': excess_ret > 0,
            })

        if not results:
            return {'error': 'No valid windows', 'n_windows': 0}

        df = pd.DataFrame(results)

        return {
            'data': df,
            'n_windows': len(df),
            'regime_days': regime_days,
            'mean_ret': df['ret_vs_baseline'].mean(),
            'median_ret': df['ret_vs_baseline'].median(),
            'std_ret': df['ret_vs_baseline'].std(),
            'p5_ret': df['ret_vs_baseline'].quantile(0.05),
            'p95_ret': df['ret_vs_baseline'].quantile(0.95),
            'prob_loss_vs_baseline': (df['ret_vs_baseline'] < 0).mean() * 100,
            'mean_excess': df['excess_ret'].mean(),
            'median_excess': df['excess_ret'].median(),
            'std_excess': df['excess_ret'].std(),
            'p5_excess': df['excess_ret'].quantile(0.05),
            'p95_excess': df['excess_ret'].quantile(0.95),
            'win_rate_vs_tbill': (df['excess_ret'] > 0).mean() * 100,
            'prob_underperform_tbill': (df['excess_ret'] < 0).mean() * 100,
            'var_95_excess': df['excess_ret'].quantile(0.05),
            'cvar_95_excess': df[df['excess_ret'] <= df['excess_ret'].quantile(0.05)]['excess_ret'].mean(),
        }


def _to_business_days(spot_series: pd.Series) -> pd.Series:
    if spot_series.index.tz is not None:
        spot_series = spot_series.tz_localize(None)
    return spot_series.asfreq('B').ffill().dropna()


def calibrate_gbm(spot_series: pd.Series, regime_days: Optional[int] = None) -> Dict:
    """Calibrate GBM parameters from business-day data."""
    spot_series = _to_business_days(spot_series)

    if regime_days is not None:
        cutoff = spot_series.index[-1] - pd.Timedelta(days=regime_days)
        spot_series = spot_series[spot_series.index >= cutoff]

    log_returns = np.log(spot_series / spot_series.shift(1)).dropna()

    if len(log_returns) < 30:
        return {'error': 'Insufficient data', 'n_returns': len(log_returns)}

    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std(ddof=1)

    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        'mu_daily': mu_daily,
        'sigma_daily': sigma_daily,
        'mu_annual': mu_annual,
        'sigma_annual': sigma_annual,
        'n_returns': len(log_returns),
        'regime_days': regime_days,
    }


def run_consistency_check(metrics: TradeMetrics) -> Dict:
    """Run internal consistency check to verify model correctness."""
    model = CarryTradeModel(metrics)

    checks = {
        'inputs': {
            'principal_try': metrics.principal_try,
            'principal_usd': metrics.principal_usd,
            'entry_bank_rate': metrics.entry_bank_rate,
            'spot_entry': metrics.spot_entry,
            'entry_spread': metrics.entry_spread,
            'term_days': metrics.term_days_calendar,
            'deposit_rate': metrics.deposit_rate_annual,
            'stopaj_rate': metrics.stopaj_rate,
            'rf_rate': metrics.usd_rf_rate_annual,
            'swift_fee': metrics.swift_fee_usd,
        },
        'computed': {
            'gross_interest': metrics.gross_interest,
            'net_interest': metrics.net_interest,
            'final_try': metrics.final_try,
            'usd0_baseline': metrics.usd0_baseline,
            'usd_rf_end': metrics.usd_rf_end,
            'rf_return_period_pct': metrics.rf_return_period * 100,
            'bank_rate_be': metrics.bank_rate_be,
            'spot_be': metrics.spot_be,
            'be_move_pct': metrics.be_move_pct,
        },
        'verification': {},
    }

    outcome_be = model.compute_outcome(metrics.spot_be)
    checks['verification']['usd_end_at_be'] = outcome_be.usd_end
    checks['verification']['excess_at_be'] = outcome_be.excess_ret_vs_tbill
    checks['verification']['be_correct'] = abs(outcome_be.excess_ret_vs_tbill) < 0.01

    test_spot_worse = metrics.spot_be * 1.02
    outcome_worse = model.compute_outcome(test_spot_worse)
    checks['verification']['worse_spot'] = test_spot_worse
    checks['verification']['worse_excess'] = outcome_worse.excess_ret_vs_tbill
    checks['verification']['direction_correct'] = outcome_worse.excess_ret_vs_tbill < 0

    test_spot_better = metrics.spot_be * 0.98
    outcome_better = model.compute_outcome(test_spot_better)
    checks['verification']['better_spot'] = test_spot_better
    checks['verification']['better_excess'] = outcome_better.excess_ret_vs_tbill
    checks['verification']['direction_correct_2'] = outcome_better.excess_ret_vs_tbill > 0

    all_correct = (
        checks['verification']['be_correct']
        and checks['verification']['direction_correct']
        and checks['verification']['direction_correct_2']
    )
    checks['verification']['ALL_CHECKS_PASS'] = all_correct

    return checks


def run_regime_comparison(
    spot_series: pd.Series,
    metrics: TradeMetrics,
    n_sims: int = 50000,
) -> pd.DataFrame:
    """Run comparison across all regime windows."""
    model = CarryTradeModel(metrics)

    regimes = [
        ('1Y', 365),
        ('2Y', 730),
        ('3Y', 1095),
        ('4Y', 1460),
        ('5Y', 1825),
    ]

    results = []

    for name, days in regimes:
        calib = calibrate_gbm(spot_series, days)

        if 'error' in calib:
            results.append({
                'regime': name,
                'error': calib['error'],
            })
            continue

        mc = model.monte_carlo(
            mu_annual=calib['mu_annual'],
            sigma_annual=calib['sigma_annual'],
            n_sims=n_sims,
        )

        bt = model.historical_backtest(spot_series, days)

        results.append({
            'regime': name,
            'regime_days': days,
            'n_returns': calib['n_returns'],
            'mu_annual': calib['mu_annual'] * 100,
            'sigma_annual': calib['sigma_annual'] * 100,
            'mc_mean_excess': mc['mean_excess'],
            'mc_median_excess': mc['median_excess'],
            'mc_std_excess': mc['std_excess'],
            'mc_p5_excess': mc['p5_excess'],
            'mc_p95_excess': mc['p95_excess'],
            'mc_prob_underperform': mc['prob_underperform_tbill'],
            'mc_var95_excess': mc['var_95_excess'],
            'mc_cvar95_excess': mc['cvar_95_excess'],
            'mc_sharpe': mc['sharpe'],
            'bt_n_windows': bt.get('n_windows', 0),
            'bt_mean_excess': bt.get('mean_excess', np.nan),
            'bt_median_excess': bt.get('median_excess', np.nan),
            'bt_win_rate': bt.get('win_rate_vs_tbill', np.nan),
            'bt_var95_excess': bt.get('var_95_excess', np.nan),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("dashboard_core loaded")
