"""
app_streamlit.py
================
Premium Streamlit Dashboard for TRY Carry Trade Analysis.

Run with:
    streamlit run app_streamlit.py
"""

import logging
import numpy as np
from datetime import datetime, date

import streamlit as st
import pandas as pd

from dashboard_core import (
    TryHolderParams,
    UsdHolderParams,
    CarryTradeModel,
    calibrate_gbm,
    run_consistency_check,
    run_regime_comparison,
    compute_try_holder_metrics,
    compute_usd_holder_metrics,
)
from data import (
    fetch_usdtry_data,
    get_spot_series,
    get_spot_at_date,
    get_regime_options,
    validate_data_sufficiency,
    get_data_summary,
)
from plots import (
    create_scenario_chart,
    create_payoff_diagram,
    create_mc_distribution,
    create_historical_chart,
    create_backtest_chart,
    create_regime_comparison_chart,
    create_verdict_html,
    create_realtime_pnl_chart,
    create_trend_chart,
    create_cushion_gauge,
    create_cushion_timeline,
    COLORS,
)
from utils import (
    format_currency,
    format_percent,
    format_number,
    generate_scenario_table_csv,
    generate_metrics_csv,
    compute_trade_economics_summary,
    print_consistency_report,
)
from realtime_analysis import (
    compute_realtime_pnl,
    compute_trend_analysis,
    compute_daily_cushion,
    generate_realtime_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="TRY Carry Trade Dashboard",
    page_icon="TRY",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #1a1d24;
    }
    .stMetric {
        background-color: #1a1d24;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
    }
    .stMetric label {
        color: #a0a0a0 !important;
    }
    .metric-positive {
        color: #00d26a !important;
    }
    .metric-negative {
        color: #ff4757 !important;
    }
    div[data-testid="stExpander"] {
        background-color: #1a1d24;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] div {
        color: #e0e0e0 !important;
    }
    div[data-testid="stExpander"] code {
        color: #74b9ff !important;
        background-color: #2d3436 !important;
    }
    pre {
        background-color: #1a1d24 !important;
        color: #74b9ff !important;
        border: 1px solid #3498db !important;
    }
    .highlight-box {
        background-color: #1a1d24;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d3436;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #fafafa !important;
    }
    .baseline-label {
        background-color: #9b59b6;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-size: 14px;
        margin-bottom: 20px;
        display: inline-block;
    }
    .stMarkdown, .stText, p, span, label {
        color: #e0e0e0 !important;
    }
    .stCodeBlock code {
        color: #74b9ff !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def load_market_data() -> pd.DataFrame:
    return fetch_usdtry_data(period="max")


def log_principal_scaling_sanity(mode: str, base_params, compute_fn, spot_entry: float) -> None:
    scale = 2.0
    if mode == "try_holder":
        scaled = TryHolderParams(**{**base_params.__dict__, "principal_try": base_params.principal_try * scale})
    else:
        scaled = UsdHolderParams(**{**base_params.__dict__, "principal_usd": base_params.principal_usd * scale})

    base_metrics = compute_fn(base_params)
    scaled_metrics = compute_fn(scaled)

    base_model = CarryTradeModel(base_metrics)
    scaled_model = CarryTradeModel(scaled_metrics)

    base_outcome = base_model.compute_outcome(spot_entry)
    scaled_outcome = scaled_model.compute_outcome(spot_entry)

    logger.info("Sanity check: principal scaling x%.1f", scale)
    logger.info("Baseline return delta: %.6f", abs(base_outcome.ret_vs_baseline - scaled_outcome.ret_vs_baseline))
    logger.info("Excess return delta: %.6f", abs(base_outcome.excess_ret_vs_tbill - scaled_outcome.excess_ret_vs_tbill))
    logger.info("P&L delta (baseline): %.2f", base_outcome.pnl_vs_baseline - scaled_outcome.pnl_vs_baseline)


def compute_regime_series(spot_series: pd.Series, lookback_days: int = 60) -> pd.DataFrame:
    spot_series = spot_series.asfreq("B").ffill().dropna()
    log_returns = np.log(spot_series / spot_series.shift(1)).dropna()
    rolling_vol = log_returns.rolling(lookback_days).std() * (252 ** 0.5) * 100
    quantiles = rolling_vol.quantile([0.33, 0.67])

    def classify(vol: float) -> str:
        if vol <= quantiles.iloc[0]:
            return "Low"
        if vol <= quantiles.iloc[1]:
            return "Medium"
        return "High"

    regimes = rolling_vol.apply(classify)
    return pd.DataFrame({"rolling_vol": rolling_vol, "regime": regimes}).dropna()


def main() -> None:
    st.title("TRY Carry Trade Analysis Dashboard")
    st.markdown(
        '<div class="baseline-label">Baseline = USD risk-free return over the same horizon</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading market data..."):
        df = load_market_data()
        spot_series = get_spot_series(df)
        data_summary = get_data_summary(df)

    with st.sidebar:
        st.header("Trade Parameters")

        mode = st.radio(
            "Mode",
            options=[
                "I START WITH TRY (TRY-holder)",
                "I START WITH USD (USD-holder)",
            ],
        )

        start_date = st.date_input("Start Date", value=date.today())
        term_days = st.number_input(
            "Term (calendar days)",
            value=32,
            min_value=7,
            max_value=365,
            step=1,
        )

        st.subheader("Rates")
        deposit_rate = st.slider(
            "Deposit Rate (% annual)",
            min_value=10.0,
            max_value=60.0,
            value=39.5,
            step=0.5,
        )
        stopaj_rate = st.slider(
            "Stopaj Tax (%)",
            min_value=0.0,
            max_value=30.0,
            value=17.5,
            step=0.5,
        )
        usd_rf_rate = st.slider(
            "US T-Bill Rate (% annual)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.25,
        )

        st.subheader("Entry FX")
        entry_date_spot, entry_spot_value = get_spot_at_date(spot_series, str(start_date))
        default_spot = entry_spot_value if entry_spot_value else float(spot_series.iloc[-1])

        spot_entry = st.number_input(
            "Spot at Entry (USD/TRY)",
            value=float(default_spot),
            min_value=30.0,
            max_value=80.0,
            step=0.01,
            format="%.4f",
        )

        if mode.startswith("I START WITH TRY"):
            principal_try = st.number_input(
                "Principal (TRY)",
                value=1_430_000,
                min_value=10_000,
                max_value=100_000_000,
                step=10_000,
                format="%d",
            )
            entry_bank_rate_try_to_usd = st.number_input(
                "Entry Bank Rate TRY to USD",
                value=43.10,
                min_value=30.0,
                max_value=80.0,
                step=0.01,
                format="%.4f",
                help="Executable rate for TRY to USD at entry",
            )
            include_swift_in_baseline = st.checkbox(
                "Subtract SWIFT fee from USD baseline",
                value=False,
            )
        else:
            principal_usd = st.number_input(
                "Principal (USD)",
                value=33000,
                min_value=1_000,
                max_value=10_000_000,
                step=500,
                format="%d",
            )
            entry_bank_rate_usd_to_try = st.number_input(
                "Entry Bank Rate USD to TRY",
                value=42.50,
                min_value=30.0,
                max_value=80.0,
                step=0.01,
                format="%.4f",
                help="Bank buy rate for USD to TRY",
            )
            include_swift_in_baseline = False

        st.subheader("Exit FX")
        exit_mode = st.radio(
            "Exit Bank Rate",
            options=["Use Spot + Exit Spread", "Manual Exit Bank Rate"],
        )
        exit_spread_pct = st.slider(
            "Exit Spread (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.75,
            step=0.05,
            disabled=exit_mode != "Use Spot + Exit Spread",
        )
        manual_exit_rate = None
        if exit_mode == "Manual Exit Bank Rate":
            manual_exit_rate = st.number_input(
                "Manual Exit Bank Rate TRY to USD",
                value=44.0,
                min_value=30.0,
                max_value=90.0,
                step=0.01,
                format="%.4f",
            )

        swift_fee = st.slider(
            "SWIFT Fee (USD)",
            min_value=0.0,
            max_value=100.0,
            value=32.50,
            step=2.5,
        )

        st.subheader("Calibration")
        regime_options = get_regime_options()
        regime_name = st.selectbox(
            "Lookback Window",
            options=list(regime_options.keys()),
            index=1,
        )
        regime_days = regime_options[regime_name]

        n_sims = st.select_slider(
            "Monte Carlo Simulations",
            options=[10000, 25000, 50000, 100000],
            value=50000,
        )

    exit_spread = exit_spread_pct / 100

    if mode.startswith("I START WITH TRY"):
        params = TryHolderParams(
            principal_try=principal_try,
            entry_bank_rate_try_to_usd=entry_bank_rate_try_to_usd,
            spot_entry=spot_entry,
            start_date=datetime.combine(start_date, datetime.min.time()),
            term_days_calendar=term_days,
            deposit_rate_annual=deposit_rate / 100,
            stopaj_rate=stopaj_rate / 100,
            swift_fee_usd=swift_fee,
            usd_rf_rate_annual=usd_rf_rate / 100,
            exit_spread=exit_spread,
            include_swift_in_baseline=include_swift_in_baseline,
            bank_rate_end_override=manual_exit_rate,
        )
        metrics = compute_try_holder_metrics(params)
        compute_fn = compute_try_holder_metrics
    else:
        params = UsdHolderParams(
            principal_usd=principal_usd,
            entry_bank_rate_usd_to_try=entry_bank_rate_usd_to_try,
            spot_entry=spot_entry,
            start_date=datetime.combine(start_date, datetime.min.time()),
            term_days_calendar=term_days,
            deposit_rate_annual=deposit_rate / 100,
            stopaj_rate=stopaj_rate / 100,
            swift_fee_usd=swift_fee,
            usd_rf_rate_annual=usd_rf_rate / 100,
            exit_spread=exit_spread,
            bank_rate_end_override=manual_exit_rate,
        )
        metrics = compute_usd_holder_metrics(params)
        compute_fn = compute_usd_holder_metrics

    model = CarryTradeModel(metrics)
    log_principal_scaling_sanity(metrics.mode, params, compute_fn, spot_entry)

    validation = validate_data_sufficiency(spot_series, regime_days)
    if not validation['sufficient']:
        st.warning(
            f"Insufficient data for {regime_name}. Using fallback: {validation['fallback_regime']} days"
        )
        regime_days = validation['fallback_regime']

    calib = calibrate_gbm(spot_series, regime_days)
    if 'error' in calib:
        st.error(f"Calibration error: {calib['error']}")
        return

    mc_results = model.monte_carlo(
        mu_annual=calib['mu_annual'],
        sigma_annual=calib['sigma_annual'],
        n_sims=n_sims,
    )

    backtest = model.historical_backtest(spot_series, regime_days)
    scenarios = model.scenario_analysis()

    st.markdown("### Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "USD Baseline",
            format_currency(metrics.usd0_baseline),
            help="Opportunity-cost baseline at entry",
        )
    with col2:
        st.metric(
            "T-Bill End Value",
            format_currency(metrics.usd_rf_end),
            f"{usd_rf_rate}% for {term_days}d",
        )
    with col3:
        st.metric(
            "Break-even Spot",
            format_number(metrics.spot_be),
            format_percent(metrics.be_move_pct),
        )
    with col4:
        st.metric(
            "Entry Spread",
            format_percent(metrics.entry_spread * 100, with_sign=True),
            "Entry bank rate vs spot",
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        delta_color = "normal" if mc_results['mean_excess'] >= 0 else "inverse"
        st.metric(
            "Expected Excess",
            format_percent(mc_results['mean_excess']),
            "vs T-Bill",
            delta_color=delta_color,
        )
    with col6:
        st.metric(
            "Sharpe Ratio",
            f"{mc_results['sharpe']:.3f}",
            "Period excess / std",
        )
    with col7:
        st.metric(
            "P(Underperform)",
            f"{mc_results['prob_underperform_tbill']:.1f}%",
            "vs T-Bill",
        )
    with col8:
        st.metric(
            "VaR 95%",
            format_percent(mc_results['var_95_excess']),
            "Excess return",
        )

    st.markdown(create_verdict_html(mc_results, metrics.__dict__), unsafe_allow_html=True)

    pnl_realtime = compute_realtime_pnl(metrics, spot_series, entry_date=str(start_date))
    trend_analysis = compute_trend_analysis(spot_series, pnl_realtime.days_remaining, pnl_realtime.current_spot)
    cushion_analysis = compute_daily_cushion(metrics, pnl_realtime.current_spot, pnl_realtime.days_remaining, trend_analysis)

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Real-Time",
        "Scenarios",
        "Payoff",
        "Monte Carlo",
        "Backtest",
        "Regimes",
    ])

    with tab0:
        st.subheader("Real-Time Position Analysis")

        spot_move = (pnl_realtime.current_spot / pnl_realtime.entry_spot - 1) * 100
        st.info(
            f"Entry: {pnl_realtime.entry_date.strftime('%m/%d/%Y')} @ {pnl_realtime.entry_spot:.4f} | "
            f"Now: {pnl_realtime.current_date.strftime('%m/%d/%Y')} @ {pnl_realtime.current_spot:.4f} "
            f"({spot_move:+.2f}%) | {pnl_realtime.days_elapsed}d elapsed, {pnl_realtime.days_remaining}d remaining"
        )

        st.markdown("### Position P&L")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        with col_p1:
            st.metric(
                "Unrealized P&L",
                f"${pnl_realtime.unrealized_pnl_usd:+,.2f}",
                f"{pnl_realtime.unrealized_return_pct:+.2f}%",
            )

        with col_p2:
            st.metric(
                "vs T-Bill",
                f"${pnl_realtime.excess_vs_tbill_now:+,.2f}",
                "Excess return",
            )

        with col_p3:
            st.metric(
                "MTM Return",
                f"{pnl_realtime.mtm_return_pct:+.2f}%",
                "At current spot",
            )

        with col_p4:
            st.metric(
                "USD Baseline",
                f"${pnl_realtime.usd0_baseline:,.2f}",
                f"@ {pnl_realtime.entry_bank_rate:.4f}",
            )

        st.markdown("### Daily Cushion Analysis")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)

        with col_c1:
            st.metric(
                "Total Cushion",
                f"{cushion_analysis.cushion_pct:.2f}%",
                f"BE @ {cushion_analysis.spot_be:.4f}",
            )

        with col_c2:
            st.metric(
                "Daily Cushion",
                f"{cushion_analysis.daily_cushion_pct:.3f}%/day",
                f"{cushion_analysis.daily_cushion_absolute:.4f} TRY",
            )

        with col_c3:
            st.metric(
                "Daily Trend",
                f"{trend_analysis.slope_pct_daily:.3f}%/day",
                f"{trend_analysis.accel_regime}",
            )

        with col_c4:
            ratio = cushion_analysis.cushion_vs_trend_ratio
            days_be = cushion_analysis.days_until_breakeven
            days_be_str = f"{days_be:.0f}d" if days_be < 1000 else "inf"
            st.metric(
                "Cushion/Trend",
                f"{ratio:.1f}x",
                f"BE in {days_be_str}",
            )

        status = cushion_analysis.status
        if "DANGER" in status:
            st.error(status)
        elif "CAUTION" in status or "WARNING" in status:
            st.warning(status)
        else:
            st.success(status)

        pnl_dict = {
            'entry_date': pnl_realtime.entry_date,
            'current_date': pnl_realtime.current_date,
            'days_elapsed': pnl_realtime.days_elapsed,
            'days_remaining': pnl_realtime.days_remaining,
            'entry_spot': pnl_realtime.entry_spot,
            'entry_bank_rate': pnl_realtime.entry_bank_rate,
            'current_spot': pnl_realtime.current_spot,
            'current_bank_rate': pnl_realtime.current_bank_rate,
            'unrealized_pnl_usd': pnl_realtime.unrealized_pnl_usd,
            'unrealized_return_pct': pnl_realtime.unrealized_return_pct,
            'excess_vs_tbill_now': pnl_realtime.excess_vs_tbill_now,
            'mtm_return_pct': pnl_realtime.mtm_return_pct,
            'spot_be': metrics.spot_be,
        }

        trend_dict = {
            'slope_daily': trend_analysis.slope_daily,
            'slope_pct_daily': trend_analysis.slope_pct_daily,
            'slope_annual': trend_analysis.slope_annual,
            'accel_regime': trend_analysis.accel_regime,
            'r_squared': trend_analysis.r_squared,
            'projected_spot_maturity': trend_analysis.projected_spot_maturity,
        }

        cushion_dict = {
            'spot_current': cushion_analysis.spot_current,
            'spot_be': cushion_analysis.spot_be,
            'cushion_absolute': cushion_analysis.cushion_absolute,
            'cushion_pct': cushion_analysis.cushion_pct,
            'days_remaining': cushion_analysis.days_remaining,
            'daily_cushion_absolute': cushion_analysis.daily_cushion_absolute,
            'daily_cushion_pct': cushion_analysis.daily_cushion_pct,
            'trend_daily_pct': cushion_analysis.trend_daily_pct,
            'cushion_vs_trend_ratio': cushion_analysis.cushion_vs_trend_ratio,
            'days_until_breakeven': cushion_analysis.days_until_breakeven,
            'status': cushion_analysis.status,
        }

        st.markdown("### Charts")

        col_rt1, col_rt2 = st.columns(2)

        with col_rt1:
            fig_pnl = create_realtime_pnl_chart(pnl_dict, spot_series)
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col_rt2:
            fig_trend = create_trend_chart(spot_series, trend_dict, pnl_realtime.days_remaining)
            st.plotly_chart(fig_trend, use_container_width=True)

        col_rt3, col_rt4 = st.columns(2)

        with col_rt3:
            fig_gauge = create_cushion_gauge(cushion_dict)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_rt4:
            fig_timeline = create_cushion_timeline(cushion_dict, trend_dict)
            st.plotly_chart(fig_timeline, use_container_width=True)

        with st.expander("Entry & Current Details"):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("Entry")
                st.text(f"Spot:        {pnl_realtime.entry_spot:.4f}")
                st.text(f"Bank Rate:   {pnl_realtime.entry_bank_rate:.4f}")
                st.text(f"USD Baseline: ${pnl_realtime.usd0_baseline:,.2f}")

            with col_d2:
                st.markdown("Current")
                st.text(f"Spot:        {pnl_realtime.current_spot:.4f}")
                st.text(f"Bank Rate:   {pnl_realtime.current_bank_rate:.4f}")
                st.text(f"TRY Value:   {pnl_realtime.current_try_value:,.2f}")
                st.text(f"USD Value:   ${pnl_realtime.current_usd_value:,.2f}")

        with st.expander("Trend Details"):
            st.text(f"1-Week Slope:    {trend_analysis.slope_1w:.4f} TRY/day")
            st.text(f"2-Week Slope:    {trend_analysis.slope_2w:.4f} TRY/day")
            st.text(f"1-Month Slope:   {trend_analysis.slope_1m:.4f} TRY/day")
            st.text(f"20-Day Slope:    {trend_analysis.slope_daily:.4f} TRY/day ({trend_analysis.slope_pct_daily:.3f}%)")
            st.text(f"Annualized:      {trend_analysis.slope_annual:.2f} TRY/year ({trend_analysis.slope_annual/pnl_realtime.current_spot*100:.1f}%)")
            st.text(f"R-squared:       {trend_analysis.r_squared:.3f}")
            st.text(f"Acceleration:    {trend_analysis.accel_daily:.4f} ({trend_analysis.accel_regime})")
            st.text(f"Projected Spot:  {trend_analysis.projected_spot_maturity:.4f} ({trend_analysis.projected_move_pct:+.2f}%)")

        with st.expander("Full Report"):
            report = generate_realtime_report(pnl_realtime, trend_analysis, cushion_analysis)
            st.code(report)

    with tab1:
        st.subheader("Scenario Analysis")
        st.markdown("P/L for different USD/TRY movements at maturity")

        fig_scenario = create_scenario_chart(scenarios, metrics.__dict__)
        st.plotly_chart(fig_scenario, use_container_width=True)

        with st.expander("Scenario Table"):
            display_df = scenarios[[
                'spot_move_pct', 'spot_end', 'bank_rate_end', 'usd_end',
                'pnl_vs_baseline', 'pnl_vs_tbill', 'ret_vs_baseline', 'excess_ret_vs_tbill'
            ]].copy()
            display_df.columns = [
                'Move %', 'Spot', 'Bank Rate', 'USD End',
                'P/L vs Baseline ($)', 'P/L vs T-Bill ($)', 'Ret vs Baseline (%)', 'Excess vs T-Bill (%)'
            ]
            st.dataframe(display_df.style.format({
                'Move %': '{:+.1f}',
                'Spot': '{:.4f}',
                'Bank Rate': '{:.4f}',
                'USD End': '${:,.2f}',
                'P/L vs Baseline ($)': '${:+,.2f}',
                'P/L vs T-Bill ($)': '${:+,.2f}',
                'Ret vs Baseline (%)': '{:+.2f}',
                'Excess vs T-Bill (%)': '{:+.2f}',
            }), use_container_width=True)

    with tab2:
        st.subheader("Payoff Diagram")
        st.markdown("Return profile across USD/TRY range")

        fig_payoff = create_payoff_diagram(metrics.__dict__)
        st.plotly_chart(fig_payoff, use_container_width=True)

    with tab3:
        st.subheader("Monte Carlo Simulation")
        st.markdown(f"{n_sims:,} simulations | mu = {calib['mu_annual']*100:.1f}% | sigma = {calib['sigma_annual']*100:.1f}%")

        col_mc1, col_mc2 = st.columns(2)

        with col_mc1:
            st.markdown("Return vs Baseline")
            fig_mc1 = create_mc_distribution(mc_results, show_excess=False)
            st.plotly_chart(fig_mc1, use_container_width=True)

        with col_mc2:
            st.markdown("Excess Returns vs T-Bill")
            fig_mc2 = create_mc_distribution(mc_results, show_excess=True)
            st.plotly_chart(fig_mc2, use_container_width=True)

        with st.expander("Monte Carlo Statistics"):
            mc_stats = pd.DataFrame({
                'Metric': [
                    'Mean Return', 'Median Return', 'Std Dev', 'P5', 'P95',
                    'VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%',
                    'P(Loss vs Baseline)', 'P(Underperform T-Bill)', 'Sharpe'
                ],
                'vs Baseline': [
                    f"{mc_results['mean_ret']:.2f}%",
                    f"{mc_results['median_ret']:.2f}%",
                    f"{mc_results['std_ret']:.2f}%",
                    f"{mc_results['p5_ret']:.2f}%",
                    f"{mc_results['p95_ret']:.2f}%",
                    f"{mc_results['var_95_ret']:.2f}%",
                    f"{mc_results['var_99_ret']:.2f}%",
                    f"{mc_results['cvar_95_ret']:.2f}%",
                    f"{mc_results['cvar_99_ret']:.2f}%",
                    f"{mc_results['prob_loss_vs_baseline']:.1f}%",
                    "-",
                    "-",
                ],
                'Excess vs T-Bill': [
                    f"{mc_results['mean_excess']:.2f}%",
                    f"{mc_results['median_excess']:.2f}%",
                    f"{mc_results['std_excess']:.2f}%",
                    f"{mc_results['p5_excess']:.2f}%",
                    f"{mc_results['p95_excess']:.2f}%",
                    f"{mc_results['var_95_excess']:.2f}%",
                    f"{mc_results['var_99_excess']:.2f}%",
                    f"{mc_results['cvar_95_excess']:.2f}%",
                    f"{mc_results['cvar_99_excess']:.2f}%",
                    "-",
                    f"{mc_results['prob_underperform_tbill']:.1f}%",
                    f"{mc_results['sharpe']:.3f}",
                ],
            })
            st.dataframe(mc_stats, use_container_width=True)

    with tab4:
        st.subheader("Historical Backtest")

        if 'error' in backtest:
            st.error(f"Backtest error: {backtest['error']}")
        else:
            st.markdown(f"{backtest['n_windows']} rolling windows from {regime_name} data")

            fig_bt = create_backtest_chart(backtest)
            st.plotly_chart(fig_bt, use_container_width=True)

            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)

            with col_bt1:
                st.metric("Win Rate vs T-Bill", f"{backtest['win_rate_vs_tbill']:.1f}%")
            with col_bt2:
                st.metric("Mean Excess", format_percent(backtest['mean_excess']))
            with col_bt3:
                st.metric("P(Underperform)", f"{backtest['prob_underperform_tbill']:.1f}%")
            with col_bt4:
                st.metric("VaR 95%", format_percent(backtest['var_95_excess']))

    with tab5:
        st.subheader("Regimes")
        st.markdown("Rolling 60-day volatility and current regime")

        regime_df = compute_regime_series(spot_series)
        if regime_df.empty:
            st.warning("Not enough data to compute 60-day volatility.")
        else:
            current_regime = regime_df['regime'].iloc[-1]
            current_vol = regime_df['rolling_vol'].iloc[-1]

            st.metric("Current Regime", current_regime, f"60d vol: {current_vol:.1f}%")

            vol_fig = {
                "data": [
                    {
                        "x": regime_df.index,
                        "y": regime_df['rolling_vol'],
                        "type": "scatter",
                        "mode": "lines",
                        "name": "Rolling 60d vol",
                        "line": {"color": COLORS['info'], "width": 2},
                    }
                ],
                "layout": {
                    "paper_bgcolor": COLORS['bg_dark'],
                    "plot_bgcolor": COLORS['bg_card'],
                    "font": {"color": COLORS['text_primary']},
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Annualized Vol (%)"},
                    "height": 350,
                },
            }
            st.plotly_chart(vol_fig, use_container_width=True)

        st.markdown("Regime comparison across lookbacks")
        with st.spinner("Running regime comparison..."):
            regime_compare = run_regime_comparison(spot_series, metrics, n_sims=min(n_sims, 25000))

        fig_regime = create_regime_comparison_chart(regime_compare)
        st.plotly_chart(fig_regime, use_container_width=True)

        st.markdown("Summary Table")
        display_cols = [
            'regime', 'n_returns', 'mu_annual', 'sigma_annual',
            'mc_mean_excess', 'mc_prob_underperform', 'mc_sharpe',
            'bt_win_rate'
        ]
        display_df = regime_compare[display_cols].copy()
        display_df.columns = [
            'Regime', 'N Returns', 'mu (% ann)', 'sigma (% ann)',
            'Mean Excess (%)', 'P(Under) %', 'Sharpe', 'BT Win Rate %'
        ]
        st.dataframe(display_df.style.format({
            'N Returns': '{:,.0f}',
            'mu (% ann)': '{:.1f}',
            'sigma (% ann)': '{:.1f}',
            'Mean Excess (%)': '{:+.2f}',
            'P(Under) %': '{:.1f}',
            'Sharpe': '{:.3f}',
            'BT Win Rate %': '{:.1f}',
        }), use_container_width=True)

    st.markdown("---")
    st.subheader("USD/TRY Historical")
    fig_hist = create_historical_chart(spot_series.tail(365), metrics.__dict__)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Export")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        csv_scenarios = generate_scenario_table_csv(scenarios)
        st.download_button(
            "Download Scenarios (CSV)",
            csv_scenarios,
            "scenarios.csv",
            "text/csv",
        )

    with col_exp2:
        csv_metrics = generate_metrics_csv(metrics.__dict__, mc_results, backtest)
        st.download_button(
            "Download Metrics (CSV)",
            csv_metrics,
            "metrics.csv",
            "text/csv",
        )

    with col_exp3:
        checks = run_consistency_check(metrics)
        report_text = compute_trade_economics_summary(metrics.__dict__)
        report_text += "\n\n" + print_consistency_report(checks)
        st.download_button(
            "Download Full Report (TXT)",
            report_text,
            "carry_trade_report.txt",
            "text/plain",
        )

    with st.expander("Methodology (optional)"):
        checks = run_consistency_check(metrics)
        st.text(print_consistency_report(checks))
        st.text(f"Data available: {data_summary['start_date']} to {data_summary['end_date']}")


if __name__ == "__main__":
    main()
