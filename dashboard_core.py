"""
dashboard_core.py
=================
Core computation module for TRY Carry Trade Analysis.

DUAL MODE SUPPORT:
==================
1) TRY Holder Mode: Start with TRY, want USD eventually
   - Baseline = Convert at entry_bank_rate_try_to_usd + T-bill
   - usd0_exec = principal_try / entry_bank_rate_try_to_usd

2) USD Holder Mode: Start with USD, convert to TRY, earn carry, convert back
   - Baseline = Keep USD in T-bill
   - try0 = principal_usd * entry_bank_rate_usd_to_try

Variables:
- usd0_exec: Executable USD at entry (TRY holder) or principal_usd (USD holder)
- usd_rf_end: T-bill end value = usd0_exec * (1 + rf * T)
- usd_end: Carry trade outcome = final_try / bank_rate_end - swift_fee
- spread: Bank spread percentage
- bank_rate_be: Break-even bank rate where carry = T-bill
- pnl_vs_tbill: usd_end - usd_rf_end (excess P/L)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODE 1: TRY HOLDER (start with TRY, want USD)
# ============================================================================

@dataclass
class TRYHolderParams:
    """
    TRY Holder Mode: User starts with TRY, wants to convert to USD eventually.

    Baseline = Convert at entry_bank_rate_try_to_usd + T-bill
    """
    # Principal
    principal_try: float = 1_430_000.0

    # Entry date and term
    entry_date: datetime = None
    term_days_calendar: int = 32

    # Deposit parameters
    deposit_rate_annual: float = 0.395  # 39.5%
    stopaj_rate: float = 0.175  # 17.5%

    # Exchange rates at entry
    spot_entry: float = 42.72  # Mid-market spot at entry (informational)
    entry_bank_rate_try_to_usd: float = 43.10  # Executable TRY->USD rate

    # Exit parameters
    exit_spread_pct: float = 0.0089  # ~0.89% spread (derived or manual)

    # Fees
    swift_fee_usd: float = 32.50

    # Benchmark
    tbill_rate_annual_usd: float = 0.035  # 3.5% T-bill

    # Options
    use_compound_interest: bool = False

    # Baseline mode: "opportunity_cost" (default) or "shadow_spot"
    baseline_mode: str = "opportunity_cost"

    # Optional overrides
    bank_rate_end_override: Optional[float] = None
    final_try_override: Optional[float] = None

    def __post_init__(self):
        """Compute derived values."""
        if self.entry_date is None:
            self.entry_date = datetime(2025, 12, 19)

        # Maturity date
        self.maturity_date = self.entry_date + timedelta(days=self.term_days_calendar)

        # Spread derived from entry rates (if spot provided)
        if self.spot_entry > 0:
            self.spread = self.entry_bank_rate_try_to_usd / self.spot_entry - 1
        else:
            self.spread = self.exit_spread_pct

        # Time fraction
        self.T = self.term_days_calendar / 365.0

        # Interest calculations
        self.gross_interest = self.principal_try * self.deposit_rate_annual * self.T
        self.stopaj_amount = self.gross_interest * self.stopaj_rate
        self.net_interest = self.gross_interest - self.stopaj_amount

        if self.final_try_override:
            self.final_try = self.final_try_override
        else:
            self.final_try = self.principal_try + self.net_interest

        # BASELINE: Opportunity-cost (convert at entry bank rate, invest in T-bill)
        self.usd0_exec = self.principal_try / self.entry_bank_rate_try_to_usd

        # T-bill end value
        if self.use_compound_interest:
            self.usd_rf_end = self.usd0_exec * (1 + self.tbill_rate_annual_usd) ** self.T
        else:
            self.usd_rf_end = self.usd0_exec * (1 + self.tbill_rate_annual_usd * self.T)

        # T-bill return (period)
        self.rf_return_period = self.usd_rf_end / self.usd0_exec - 1

        # Break-even bank rate (where carry = T-bill)
        # Solve: final_try / bank_rate_be - swift = usd_rf_end
        self.bank_rate_be = self.final_try / (self.usd_rf_end + self.swift_fee_usd)
        self.spot_be = self.bank_rate_be / (1 + self.spread)

        # Break-even move from entry spot
        self.be_move_pct = (self.spot_be / self.spot_entry - 1) * 100

        # Legacy compatibility
        self.entry_bank_rate = self.entry_bank_rate_try_to_usd
        self.usd_rf_rate_annual = self.tbill_rate_annual_usd


# ============================================================================
# MODE 2: USD HOLDER (start with USD, convert to TRY, earn carry, convert back)
# ============================================================================

@dataclass
class USDHolderParams:
    """
    USD Holder Mode: User starts with USD, converts to TRY for carry, converts back.

    Baseline = Keep USD in T-bill
    """
    # Principal
    principal_usd: float = 33_000.0

    # Entry date and term
    entry_date: datetime = None
    term_days_calendar: int = 32

    # Deposit parameters
    deposit_rate_annual: float = 0.395  # 39.5%
    stopaj_rate: float = 0.175  # 17.5%

    # Exchange rates at entry
    spot_entry: float = 42.72  # Mid-market spot at entry (informational)
    entry_bank_rate_usd_to_try: float = 42.34  # Executable USD->TRY rate (bank haircut)

    # Exit parameters
    exit_spread_pct: float = 0.0089  # ~0.89% spread at exit

    # Fees
    swift_fee_usd: float = 32.50

    # Benchmark
    tbill_rate_annual_usd: float = 0.035  # 3.5% T-bill

    # Options
    use_compound_interest: bool = False

    # Optional overrides
    bank_rate_end_override: Optional[float] = None

    def __post_init__(self):
        """Compute derived values."""
        if self.entry_date is None:
            self.entry_date = datetime(2025, 12, 19)

        # Maturity date
        self.maturity_date = self.entry_date + timedelta(days=self.term_days_calendar)

        # Convert USD to TRY at entry
        self.try0 = self.principal_usd * self.entry_bank_rate_usd_to_try
        self.principal_try = self.try0  # For compatibility

        # Spread for exit (derive from spot or use manual)
        if self.spot_entry > 0:
            self.spread = self.exit_spread_pct
        else:
            self.spread = self.exit_spread_pct

        # Time fraction
        self.T = self.term_days_calendar / 365.0

        # Interest calculations on TRY
        self.gross_interest = self.try0 * self.deposit_rate_annual * self.T
        self.stopaj_amount = self.gross_interest * self.stopaj_rate
        self.net_interest = self.gross_interest - self.stopaj_amount
        self.final_try = self.try0 + self.net_interest

        # BASELINE: T-bill on original USD
        self.usd0_exec = self.principal_usd  # The USD you started with

        # T-bill end value
        if self.use_compound_interest:
            self.usd_rf_end = self.usd0_exec * (1 + self.tbill_rate_annual_usd) ** self.T
        else:
            self.usd_rf_end = self.usd0_exec * (1 + self.tbill_rate_annual_usd * self.T)

        # T-bill return (period)
        self.rf_return_period = self.usd_rf_end / self.usd0_exec - 1

        # Break-even: solve for spot_exit where usd_end = usd_rf_end
        # usd_end = final_try / (spot_exit * (1 + spread)) - swift = usd_rf_end
        # spot_exit * (1 + spread) = final_try / (usd_rf_end + swift)
        self.bank_rate_be = self.final_try / (self.usd_rf_end + self.swift_fee_usd)
        self.spot_be = self.bank_rate_be / (1 + self.spread)

        # Break-even move from entry spot
        self.be_move_pct = (self.spot_be / self.spot_entry - 1) * 100

        # Legacy compatibility
        self.entry_bank_rate = self.spot_entry * (1 + self.spread)  # For exit calc
        self.usd_rf_rate_annual = self.tbill_rate_annual_usd


# ============================================================================
# LEGACY COMPATIBILITY: TradeParams (maps to TRY Holder)
# ============================================================================

@dataclass
class TradeParams:
    """
    Trade parameters - LEGACY COMPATIBILITY wrapper.
    Maps to TRYHolderParams for backward compatibility.
    """
    principal_try: float = 1_430_000.0
    deposit_rate_annual: float = 0.395  # 39.5%
    term_days_calendar: int = 32
    stopaj_rate: float = 0.175  # 17.5%
    entry_bank_rate: float = 43.10  # TRY per USD (executable rate)
    spot_entry: float = 42.72  # Mid-market spot at entry
    swift_fee_usd: float = 32.50
    usd_rf_rate_annual: float = 0.035  # 3.5% T-bill
    use_compound_interest: bool = False  # Simple interest by default

    # Optional: Direct bank quote at maturity (overrides spread calculation)
    bank_rate_end_override: Optional[float] = None

    # New: entry date support
    entry_date: datetime = None

    def __post_init__(self):
        """Compute derived values."""
        if self.entry_date is None:
            self.entry_date = datetime(2025, 12, 19)

        # Maturity date
        self.maturity_date = self.entry_date + timedelta(days=self.term_days_calendar)

        # Spread derived from entry rates
        self.spread = self.entry_bank_rate / self.spot_entry - 1

        # Time fraction
        self.T = self.term_days_calendar / 365.0

        # Interest calculations
        self.gross_interest = self.principal_try * self.deposit_rate_annual * self.T
        self.stopaj_amount = self.gross_interest * self.stopaj_rate
        self.net_interest = self.gross_interest - self.stopaj_amount
        self.final_try = self.principal_try + self.net_interest

        # Opportunity-cost baseline (EXECUTABLE USD at entry)
        self.usd0_exec = self.principal_try / self.entry_bank_rate

        # T-bill end value (risk-free comparator)
        if self.use_compound_interest:
            self.usd_rf_end = self.usd0_exec * (1 + self.usd_rf_rate_annual) ** self.T
        else:
            self.usd_rf_end = self.usd0_exec * (1 + self.usd_rf_rate_annual * self.T)

        # T-bill return (period)
        self.rf_return_period = self.usd_rf_end / self.usd0_exec - 1

        # Break-even bank rate (where carry = T-bill)
        # Solve: final_try / bank_rate_be - swift = usd_rf_end
        # => bank_rate_be = final_try / (usd_rf_end + swift)
        self.bank_rate_be = self.final_try / (self.usd_rf_end + self.swift_fee_usd)
        self.spot_be = self.bank_rate_be / (1 + self.spread)

        # Break-even move from entry spot
        self.be_move_pct = (self.spot_be / self.spot_entry - 1) * 100

        # Legacy compatibility
        self.tbill_rate_annual_usd = self.usd_rf_rate_annual
        self.entry_bank_rate_try_to_usd = self.entry_bank_rate


@dataclass
class CarryOutcome:
    """Outcome of carry trade at a given maturity spot."""
    spot_end: float
    bank_rate_end: float
    usd_end: float
    pnl_vs_convert_now: float  # vs usd0_exec
    pnl_vs_tbill: float  # vs usd_rf_end (excess P/L)
    ret_vs_convert_now: float  # percentage
    excess_ret_vs_tbill: float  # percentage
    beats_tbill: bool


class CarryTradeModel:
    """
    Correctly models TRY carry trade with opportunity-cost baseline.

    Baseline = EXECUTABLE USD at entry (not mid-spot shadow).
    Benchmark = T-bill return on that executable USD.
    """

    def __init__(self, params: TradeParams):
        self.params = params
        self.p = params  # Shorthand

    def compute_outcome(self, spot_end: float, bank_rate_end: Optional[float] = None) -> CarryOutcome:
        """
        Compute carry trade outcome for given maturity spot.

        Args:
            spot_end: Spot rate at maturity
            bank_rate_end: Bank's executable rate at maturity (optional, else derived from spread)

        Returns:
            CarryOutcome with all metrics
        """
        p = self.p

        # Bank rate at maturity
        if bank_rate_end is not None:
            br_end = bank_rate_end
        elif p.bank_rate_end_override is not None:
            br_end = p.bank_rate_end_override
        else:
            # Derive from spot using constant spread
            br_end = spot_end * (1 + p.spread)

        # USD received at maturity
        usd_end = p.final_try / br_end - p.swift_fee_usd

        # P/L vs converting now (at entry)
        pnl_vs_convert_now = usd_end - p.usd0_exec

        # P/L vs T-bill (excess return)
        pnl_vs_tbill = usd_end - p.usd_rf_end

        # Returns (percentage)
        ret_vs_convert_now = (usd_end / p.usd0_exec - 1) * 100
        excess_ret_vs_tbill = (usd_end / p.usd_rf_end - 1) * 100

        return CarryOutcome(
            spot_end=spot_end,
            bank_rate_end=br_end,
            usd_end=usd_end,
            pnl_vs_convert_now=pnl_vs_convert_now,
            pnl_vs_tbill=pnl_vs_tbill,
            ret_vs_convert_now=ret_vs_convert_now,
            excess_ret_vs_tbill=excess_ret_vs_tbill,
            beats_tbill=pnl_vs_tbill > 0
        )

    def scenario_analysis(self, spot_moves_pct: List[float] = None) -> pd.DataFrame:
        """
        Run scenario analysis for various spot movements.

        Args:
            spot_moves_pct: List of spot movements in % (default: standard range)

        Returns:
            DataFrame with scenario results
        """
        if spot_moves_pct is None:
            # Use 0.5% increments from -10% to +10%, include break-even
            be = self.p.be_move_pct
            spot_moves_pct = [x / 2 for x in range(-20, 21)]  # -10 to +10 in 0.5% steps
            spot_moves_pct.append(round(be, 2))  # Include break-even
            spot_moves_pct = sorted(set(spot_moves_pct))

        results = []
        for move in spot_moves_pct:
            spot_end = self.p.spot_entry * (1 + move / 100)
            outcome = self.compute_outcome(spot_end)

            results.append({
                'spot_move_pct': move,
                'spot_end': spot_end,
                'bank_rate_end': outcome.bank_rate_end,
                'usd_end': outcome.usd_end,
                'pnl_vs_convert_now': outcome.pnl_vs_convert_now,
                'pnl_vs_tbill': outcome.pnl_vs_tbill,
                'ret_vs_convert_now': outcome.ret_vs_convert_now,
                'excess_ret_vs_tbill': outcome.excess_ret_vs_tbill,
                'beats_tbill': outcome.beats_tbill,
                'is_breakeven': abs(move - self.p.be_move_pct) < 0.1,
            })

        return pd.DataFrame(results)

    def monte_carlo(
        self,
        mu_annual: float,
        sigma_annual: float,
        spot_start: float = None,
        n_sims: int = 50000,
        seed: int = 42
    ) -> Dict:
        """
        Monte Carlo simulation using GBM with calendar time.

        Args:
            mu_annual: Annualized drift (from calibration)
            sigma_annual: Annualized volatility (from calibration)
            spot_start: Starting spot (default: current spot from params)
            n_sims: Number of simulations
            seed: Random seed

        Returns:
            Dict with simulation results and statistics
        """
        p = self.p

        if spot_start is None:
            spot_start = p.spot_entry

        T = p.T  # Calendar time in years

        # GBM simulation
        np.random.seed(seed)
        Z = np.random.standard_normal(n_sims)

        drift = (mu_annual - 0.5 * sigma_annual**2) * T
        diffusion = sigma_annual * np.sqrt(T) * Z

        spot_ends = spot_start * np.exp(drift + diffusion)

        # Compute outcomes for all paths
        ret_vs_convert = np.zeros(n_sims)
        excess_ret = np.zeros(n_sims)
        usd_ends = np.zeros(n_sims)

        for i, spot_end in enumerate(spot_ends):
            outcome = self.compute_outcome(spot_end)
            ret_vs_convert[i] = outcome.ret_vs_convert_now
            excess_ret[i] = outcome.excess_ret_vs_tbill
            usd_ends[i] = outcome.usd_end

        # Statistics
        prob_underperform_tbill = (excess_ret < 0).mean() * 100
        prob_loss_vs_convert = (ret_vs_convert < 0).mean() * 100

        # VaR and CVaR on excess returns
        var_95_excess = np.percentile(excess_ret, 5)
        var_99_excess = np.percentile(excess_ret, 1)
        cvar_95_excess = excess_ret[excess_ret <= var_95_excess].mean()
        cvar_99_excess = excess_ret[excess_ret <= var_99_excess].mean()

        # VaR and CVaR on absolute returns
        var_95_ret = np.percentile(ret_vs_convert, 5)
        var_99_ret = np.percentile(ret_vs_convert, 1)
        cvar_95_ret = ret_vs_convert[ret_vs_convert <= var_95_ret].mean()
        cvar_99_ret = ret_vs_convert[ret_vs_convert <= var_99_ret].mean()

        # Sharpe ratio on period excess returns
        sharpe = excess_ret.mean() / excess_ret.std() if excess_ret.std() > 0 else 0

        return {
            'spot_ends': spot_ends,
            'usd_ends': usd_ends,
            'ret_vs_convert': ret_vs_convert,
            'excess_ret': excess_ret,
            'n_sims': n_sims,
            'mu_annual': mu_annual,
            'sigma_annual': sigma_annual,
            # Summary stats - Returns vs Convert Now
            'mean_ret': ret_vs_convert.mean(),
            'median_ret': np.median(ret_vs_convert),
            'std_ret': ret_vs_convert.std(),
            'p5_ret': np.percentile(ret_vs_convert, 5),
            'p95_ret': np.percentile(ret_vs_convert, 95),
            'var_95_ret': var_95_ret,
            'var_99_ret': var_99_ret,
            'cvar_95_ret': cvar_95_ret,
            'cvar_99_ret': cvar_99_ret,
            'prob_loss_vs_convert': prob_loss_vs_convert,
            # Summary stats - Excess vs T-Bill
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
            # Sharpe
            'sharpe': sharpe,
            # Dollar amounts
            'mean_usd_end': usd_ends.mean(),
            'mean_pnl_vs_convert': usd_ends.mean() - p.usd0_exec,
            'mean_pnl_vs_tbill': usd_ends.mean() - p.usd_rf_end,
        }

    def historical_backtest(
        self,
        spot_series: pd.Series,
        regime_days: int = None
    ) -> Dict:
        """
        Historical backtest using rolling calendar-day windows.

        For each start date t in regime window:
        - Derive entry bank rate from spot using spread
        - Find end date = first available date >= start + term_days (calendar)
        - Compute outcomes

        Args:
            spot_series: DatetimeIndex -> spot rate
            regime_days: Use only last N calendar days (None = full history)

        Returns:
            Dict with backtest results
        """
        p = self.p

        # Apply regime filter
        if regime_days is not None:
            cutoff = spot_series.index[-1] - pd.Timedelta(days=regime_days)
            spot_series = spot_series[spot_series.index >= cutoff]

        if len(spot_series) < 60:
            return {'error': 'Insufficient data', 'n_windows': 0}

        results = []
        dates = spot_series.index

        for i, start_date in enumerate(dates):
            # Target end date (calendar days)
            target_end = start_date + pd.Timedelta(days=p.term_days_calendar)

            # Find first available date >= target
            future_dates = dates[dates >= target_end]
            if len(future_dates) == 0:
                continue

            end_date = future_dates[0]

            spot_start = spot_series[start_date]
            spot_end = spot_series[end_date]

            # Entry bank rate for this window (using constant spread)
            bank_rate_entry_t = spot_start * (1 + p.spread)

            # Executable USD at entry for this window
            usd0_exec_t = p.principal_try / bank_rate_entry_t

            # T-bill end value for this window
            if p.use_compound_interest:
                usd_rf_end_t = usd0_exec_t * (1 + p.usd_rf_rate_annual) ** p.T
            else:
                usd_rf_end_t = usd0_exec_t * (1 + p.usd_rf_rate_annual * p.T)

            # Bank rate at maturity
            bank_rate_end_t = spot_end * (1 + p.spread)

            # USD at maturity
            usd_end_t = p.final_try / bank_rate_end_t - p.swift_fee_usd

            # Metrics
            pnl_vs_convert = usd_end_t - usd0_exec_t
            pnl_vs_tbill = usd_end_t - usd_rf_end_t
            ret_vs_convert = (usd_end_t / usd0_exec_t - 1) * 100
            excess_ret = (usd_end_t / usd_rf_end_t - 1) * 100

            results.append({
                'start_date': start_date,
                'end_date': end_date,
                'spot_start': spot_start,
                'spot_end': spot_end,
                'spot_move_pct': (spot_end / spot_start - 1) * 100,
                'usd0_exec_t': usd0_exec_t,
                'usd_rf_end_t': usd_rf_end_t,
                'usd_end_t': usd_end_t,
                'pnl_vs_convert': pnl_vs_convert,
                'pnl_vs_tbill': pnl_vs_tbill,
                'ret_vs_convert': ret_vs_convert,
                'excess_ret': excess_ret,
                'beats_tbill': excess_ret > 0,
            })

        if not results:
            return {'error': 'No valid windows', 'n_windows': 0}

        df = pd.DataFrame(results)

        # Summary statistics
        return {
            'data': df,
            'n_windows': len(df),
            'regime_days': regime_days,
            # Returns vs Convert Now
            'mean_ret': df['ret_vs_convert'].mean(),
            'median_ret': df['ret_vs_convert'].median(),
            'std_ret': df['ret_vs_convert'].std(),
            'p5_ret': df['ret_vs_convert'].quantile(0.05),
            'p95_ret': df['ret_vs_convert'].quantile(0.95),
            'prob_loss_vs_convert': (df['ret_vs_convert'] < 0).mean() * 100,
            # Excess vs T-Bill
            'mean_excess': df['excess_ret'].mean(),
            'median_excess': df['excess_ret'].median(),
            'std_excess': df['excess_ret'].std(),
            'p5_excess': df['excess_ret'].quantile(0.05),
            'p95_excess': df['excess_ret'].quantile(0.95),
            'win_rate_vs_tbill': (df['excess_ret'] > 0).mean() * 100,
            'prob_underperform_tbill': (df['excess_ret'] < 0).mean() * 100,
            # VaR/CVaR
            'var_95_excess': df['excess_ret'].quantile(0.05),
            'cvar_95_excess': df[df['excess_ret'] <= df['excess_ret'].quantile(0.05)]['excess_ret'].mean(),
        }


def calibrate_gbm(
    spot_series: pd.Series,
    regime_days: int = None
) -> Dict:
    """
    Calibrate GBM parameters from historical data.

    CRITICAL: Use BUSINESS DAYS (252) for FX drift/vol estimation.
    Deposit/T-bill accrual uses 365, but FX estimation uses trading days.

    Args:
        spot_series: DatetimeIndex -> spot rate (business day data from yfinance)
        regime_days: Use only last N calendar days for filtering

    Returns:
        Dict with mu_daily, sigma_daily, mu_annual, sigma_annual, n_returns
    """
    # Apply regime filter (calendar days for lookback window)
    if regime_days is not None:
        cutoff = spot_series.index[-1] - pd.Timedelta(days=regime_days)
        spot_series = spot_series[spot_series.index >= cutoff]

    # Use business-day log returns (raw yfinance data is already business days)
    log_returns = np.log(spot_series / spot_series.shift(1)).dropna()

    if len(log_returns) < 30:
        return {'error': 'Insufficient data', 'n_returns': len(log_returns)}

    # Daily stats (per trading day)
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std(ddof=1)

    # Annualize using 252 trading days (standard for FX)
    annualization_factor = 252
    mu_annual = mu_daily * annualization_factor
    sigma_annual = sigma_daily * np.sqrt(annualization_factor)

    return {
        'mu_daily': mu_daily,
        'sigma_daily': sigma_daily,
        'mu_annual': mu_annual,
        'sigma_annual': sigma_annual,
        'n_returns': len(log_returns),
        'regime_days': regime_days,
        'annualization_factor': annualization_factor,
    }


def compute_rolling_volatility(spot_series: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling volatility and regime classification.

    Args:
        spot_series: DatetimeIndex -> spot rate
        window: Rolling window in trading days (default 60)

    Returns:
        DataFrame with rolling_vol, regime columns
    """
    log_returns = np.log(spot_series / spot_series.shift(1)).dropna()

    # Rolling volatility (annualized)
    rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252) * 100

    # Regime classification based on historical quantiles
    vol_33 = rolling_vol.quantile(0.33)
    vol_67 = rolling_vol.quantile(0.67)

    def classify_regime(vol):
        if pd.isna(vol):
            return None
        elif vol <= vol_33:
            return 'LOW'
        elif vol <= vol_67:
            return 'MEDIUM'
        else:
            return 'HIGH'

    regime = rolling_vol.apply(classify_regime)

    return pd.DataFrame({
        'rolling_vol': rolling_vol,
        'regime': regime,
        'vol_33_threshold': vol_33,
        'vol_67_threshold': vol_67,
    })


def run_consistency_check(params: TradeParams = None) -> Dict:
    """
    Run internal consistency check to verify model correctness.

    Uses default params or provided params to verify:
    1. usd0_exec calculation
    2. usd_rf_end (simple interest)
    3. bank_rate_be calculation
    4. Break-even direction logic

    Returns:
        Dict with check results
    """
    if params is None:
        params = TradeParams()

    model = CarryTradeModel(params)
    p = params

    checks = {
        'inputs': {
            'principal_try': p.principal_try,
            'entry_bank_rate': p.entry_bank_rate,
            'spot_entry': p.spot_entry,
            'spread': p.spread,
            'term_days': p.term_days_calendar,
            'deposit_rate': p.deposit_rate_annual,
            'stopaj_rate': p.stopaj_rate,
            'rf_rate': p.usd_rf_rate_annual,
            'swift_fee': p.swift_fee_usd,
        },
        'computed': {
            'gross_interest': p.gross_interest,
            'net_interest': p.net_interest,
            'final_try': p.final_try,
            'usd0_exec': p.usd0_exec,
            'usd_rf_end': p.usd_rf_end,
            'rf_return_period_pct': p.rf_return_period * 100,
            'bank_rate_be': p.bank_rate_be,
            'spot_be': p.spot_be,
            'be_move_pct': p.be_move_pct,
        },
        'verification': {},
    }

    # Verify break-even logic
    # At break-even: carry = T-bill
    outcome_be = model.compute_outcome(p.spot_be)
    checks['verification']['usd_end_at_be'] = outcome_be.usd_end
    checks['verification']['excess_at_be'] = outcome_be.excess_ret_vs_tbill
    checks['verification']['be_correct'] = abs(outcome_be.excess_ret_vs_tbill) < 0.01

    # Verify direction: if bank_rate > bank_rate_be, should underperform
    test_spot_worse = p.spot_be * 1.02  # 2% higher than BE
    outcome_worse = model.compute_outcome(test_spot_worse)
    checks['verification']['worse_spot'] = test_spot_worse
    checks['verification']['worse_excess'] = outcome_worse.excess_ret_vs_tbill
    checks['verification']['direction_correct'] = outcome_worse.excess_ret_vs_tbill < 0

    # Verify direction: if bank_rate < bank_rate_be, should outperform
    test_spot_better = p.spot_be * 0.98  # 2% lower than BE
    outcome_better = model.compute_outcome(test_spot_better)
    checks['verification']['better_spot'] = test_spot_better
    checks['verification']['better_excess'] = outcome_better.excess_ret_vs_tbill
    checks['verification']['direction_correct_2'] = outcome_better.excess_ret_vs_tbill > 0

    # Final verdict
    all_correct = (
        checks['verification']['be_correct'] and
        checks['verification']['direction_correct'] and
        checks['verification']['direction_correct_2']
    )
    checks['verification']['ALL_CHECKS_PASS'] = all_correct

    return checks


def run_regime_comparison(
    spot_series: pd.Series,
    params: TradeParams = None,
    n_sims: int = 50000
) -> pd.DataFrame:
    """
    Run comparison across all regime windows.

    Returns DataFrame with columns:
    - lookback, mu, sigma, expected_excess_pct, p_underperform, VaR95_excess, CVaR95_excess, break_even_spot

    Note: "Underperform" = carry_end < tbill_end (NOT just < 0 return)
    """
    if params is None:
        params = TradeParams()

    model = CarryTradeModel(params)

    regimes = [
        ('1Y', 365),
        ('2Y', 730),
        ('3Y', 1095),
        ('4Y', 1460),
        ('5Y', 1825),
        ('MAX', None),
    ]

    results = []

    for name, days in regimes:
        # Calibrate GBM with 252-day annualization for FX
        calib = calibrate_gbm(spot_series, days)

        if 'error' in calib:
            results.append({
                'regime': name,
                'regime_days': days,
                'error': calib['error'],
            })
            continue

        # Monte Carlo
        mc = model.monte_carlo(
            mu_annual=calib['mu_annual'],
            sigma_annual=calib['sigma_annual'],
            n_sims=n_sims
        )

        # Historical backtest
        bt = model.historical_backtest(spot_series, days)

        results.append({
            'regime': name,
            'regime_days': days,
            'n_returns': calib['n_returns'],
            # Calibrated parameters (as %)
            'mu_annual': calib['mu_annual'] * 100,
            'sigma_annual': calib['sigma_annual'] * 100,
            # Key outputs for comparison table
            'expected_excess_pct': mc['mean_excess'],
            'p_underperform': mc['prob_underperform_tbill'],
            'var95_excess': mc['var_95_excess'],
            'cvar95_excess': mc['cvar_95_excess'],
            'break_even_spot': params.spot_be,
            # Full MC results
            'mc_mean_excess': mc['mean_excess'],
            'mc_median_excess': mc['median_excess'],
            'mc_std_excess': mc['std_excess'],
            'mc_p5_excess': mc['p5_excess'],
            'mc_p95_excess': mc['p95_excess'],
            'mc_prob_underperform': mc['prob_underperform_tbill'],
            'mc_var95_excess': mc['var_95_excess'],
            'mc_cvar95_excess': mc['cvar_95_excess'],
            'mc_sharpe': mc['sharpe'],
            # Backtest results
            'bt_n_windows': bt.get('n_windows', 0),
            'bt_mean_excess': bt.get('mean_excess', np.nan),
            'bt_median_excess': bt.get('median_excess', np.nan),
            'bt_win_rate': bt.get('win_rate_vs_tbill', np.nan),
            'bt_var95_excess': bt.get('var_95_excess', np.nan),
        })

    return pd.DataFrame(results)


def get_trade_status(params, current_date: datetime = None) -> Dict:
    """
    Determine trade status based on current date vs maturity.

    Returns:
        Dict with status, days_elapsed, days_remaining, is_matured
    """
    if current_date is None:
        current_date = datetime.now()

    entry_date = params.entry_date
    maturity_date = params.maturity_date

    days_elapsed = (current_date - entry_date).days
    days_remaining = (maturity_date - current_date).days

    if current_date >= maturity_date:
        status = "MATURED"
        is_matured = True
    elif current_date < entry_date:
        status = "NOT_STARTED"
        is_matured = False
    else:
        status = "ACTIVE"
        is_matured = False

    return {
        'status': status,
        'entry_date': entry_date,
        'maturity_date': maturity_date,
        'current_date': current_date,
        'days_elapsed': max(0, days_elapsed),
        'days_remaining': max(0, days_remaining),
        'is_matured': is_matured,
        'progress_pct': min(100, max(0, days_elapsed / params.term_days_calendar * 100)) if params.term_days_calendar > 0 else 0,
    }


def compute_try_holder_metrics(
    principal_try: float,
    deposit_rate: float,
    term_days: int,
    stopaj_rate: float,
    entry_bank_rate: float,
    spot_entry: float,
    swift_fee: float,
    tbill_rate: float,
    spot_at_maturity: float = None,
) -> Dict:
    """
    Convenience function to compute TRY holder mode metrics.

    TRY Holder: Starts with TRY, wants USD eventually.
    Baseline = Convert at entry bank rate + T-bill

    Args:
        principal_try: TRY principal amount
        deposit_rate: Annual deposit rate (decimal, e.g., 0.395)
        term_days: Term in calendar days
        stopaj_rate: Stopaj tax rate (decimal, e.g., 0.175)
        entry_bank_rate: Bank rate TRY→USD at entry
        spot_entry: Mid-market spot at entry
        swift_fee: SWIFT fee in USD
        tbill_rate: US T-bill rate (decimal, e.g., 0.035)
        spot_at_maturity: Optional spot for outcome calculation

    Returns:
        Dict with all key metrics
    """
    params = TRYHolderParams(
        principal_try=principal_try,
        deposit_rate_annual=deposit_rate,
        term_days_calendar=term_days,
        stopaj_rate=stopaj_rate,
        entry_bank_rate_try_to_usd=entry_bank_rate,
        spot_entry=spot_entry,
        swift_fee_usd=swift_fee,
        tbill_rate_annual_usd=tbill_rate,
    )

    metrics = {
        # Inputs
        'mode': 'TRY_HOLDER',
        'principal_try': principal_try,
        'deposit_rate': deposit_rate,
        'term_days': term_days,
        # Interest
        'gross_interest': params.gross_interest,
        'stopaj': params.stopaj_amount,
        'net_interest': params.net_interest,
        'final_try': params.final_try,
        # Baseline
        'usd0_exec': params.usd0_exec,
        'usd_rf_end': params.usd_rf_end,
        'spread': params.spread,
        # Break-even
        'bank_rate_be': params.bank_rate_be,
        'spot_be': params.spot_be,
        'be_move_pct': params.be_move_pct,
    }

    if spot_at_maturity is not None:
        model = CarryTradeModel(params)
        outcome = model.compute_outcome(spot_at_maturity)
        metrics.update({
            'spot_at_maturity': spot_at_maturity,
            'usd_end': outcome.usd_end,
            'pnl_vs_tbill': outcome.pnl_vs_tbill,
            'excess_ret_pct': outcome.excess_ret_vs_tbill,
        })

    return metrics


def compute_usd_holder_metrics(
    principal_usd: float,
    deposit_rate: float,
    term_days: int,
    stopaj_rate: float,
    entry_bank_rate_usd_to_try: float,
    spot_entry: float,
    exit_spread_pct: float,
    swift_fee: float,
    tbill_rate: float,
    spot_at_maturity: float = None,
) -> Dict:
    """
    Convenience function to compute USD holder mode metrics.

    USD Holder: Starts with USD, converts to TRY, earns carry, converts back.
    Baseline = Keep USD in T-bill

    Args:
        principal_usd: USD principal amount
        deposit_rate: Annual TRY deposit rate (decimal)
        term_days: Term in calendar days
        stopaj_rate: Stopaj tax rate (decimal)
        entry_bank_rate_usd_to_try: Bank rate USD→TRY at entry
        spot_entry: Mid-market spot at entry
        exit_spread_pct: Exit spread percentage (decimal)
        swift_fee: SWIFT fee in USD
        tbill_rate: US T-bill rate (decimal)
        spot_at_maturity: Optional spot for outcome calculation

    Returns:
        Dict with all key metrics
    """
    params = USDHolderParams(
        principal_usd=principal_usd,
        deposit_rate_annual=deposit_rate,
        term_days_calendar=term_days,
        stopaj_rate=stopaj_rate,
        entry_bank_rate_usd_to_try=entry_bank_rate_usd_to_try,
        spot_entry=spot_entry,
        exit_spread_pct=exit_spread_pct,
        swift_fee_usd=swift_fee,
        tbill_rate_annual_usd=tbill_rate,
    )

    metrics = {
        # Inputs
        'mode': 'USD_HOLDER',
        'principal_usd': principal_usd,
        'deposit_rate': deposit_rate,
        'term_days': term_days,
        # Entry conversion
        'try0': params.try0,
        # Interest
        'gross_interest': params.gross_interest,
        'stopaj': params.stopaj_amount,
        'net_interest': params.net_interest,
        'final_try': params.final_try,
        # Baseline (T-bill on original USD)
        'usd0_exec': params.usd0_exec,
        'usd_rf_end': params.usd_rf_end,
        'spread': params.spread,
        # Break-even
        'bank_rate_be': params.bank_rate_be,
        'spot_be': params.spot_be,
        'be_move_pct': params.be_move_pct,
    }

    if spot_at_maturity is not None:
        model = CarryTradeModel(params)
        outcome = model.compute_outcome(spot_at_maturity)
        metrics.update({
            'spot_at_maturity': spot_at_maturity,
            'usd_end': outcome.usd_end,
            'pnl_vs_tbill': outcome.pnl_vs_tbill,
            'excess_ret_pct': outcome.excess_ret_vs_tbill,
        })

    return metrics


if __name__ == "__main__":
    # Run consistency check
    print("=" * 60)
    print("INTERNAL CONSISTENCY CHECK")
    print("=" * 60)

    checks = run_consistency_check()

    print("\nINPUTS:")
    for k, v in checks['inputs'].items():
        print(f"  {k}: {v}")

    print("\nCOMPUTED VALUES:")
    for k, v in checks['computed'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nVERIFICATION:")
    for k, v in checks['verification'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.4f}")
        elif isinstance(v, bool):
            status = "PASS" if v else "FAIL"
            print(f"  {k}: {status}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    if checks['verification']['ALL_CHECKS_PASS']:
        print("ALL CONSISTENCY CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - REVIEW MODEL")
    print("=" * 60)
