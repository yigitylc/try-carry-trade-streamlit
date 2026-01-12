"""
app_streamlit.py
================
Premium Streamlit Dashboard for TRY Carry Trade Analysis.

Run with:
    streamlit run app_streamlit.py

Features:
- Dark theme with premium styling
- Interactive parameter adjustment
- Real-time recalculation
- Regime-based analysis
- Export to HTML/CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# Import local modules
from dashboard_core import TradeParams, CarryTradeModel, calibrate_gbm, run_consistency_check, run_regime_comparison
from data import fetch_usdtry_data, get_spot_series, get_spot_at_date, get_regime_options, validate_data_sufficiency, get_data_summary
from plots import (
    create_scenario_chart, create_payoff_diagram, create_mc_distribution,
    create_historical_chart, create_backtest_chart, create_regime_comparison_chart,
    create_kpi_cards_html, create_verdict_html, export_charts_to_html, COLORS,
    create_realtime_pnl_chart, create_trend_chart, create_cushion_gauge,
    create_cushion_timeline, create_realtime_kpi_html
)
from utils import (
    format_currency, format_percent, format_number, validate_params,
    generate_scenario_table_csv, generate_metrics_csv, compute_trade_economics_summary,
    print_consistency_report
)
from realtime_analysis import (
    compute_realtime_pnl, compute_trend_analysis, compute_daily_cushion,
    compute_adjusted_expected_return, generate_realtime_report
)

# Page config
st.set_page_config(
    page_title="TRY Carry Trade Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
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
    /* Fix text visibility in all contexts */
    .stMarkdown, .stText, p, span, label {
        color: #e0e0e0 !important;
    }
    /* Ensure code blocks are visible */
    .stCodeBlock code {
        color: #74b9ff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_market_data():
    """Load and cache market data."""
    df = fetch_usdtry_data(period="max")
    return df


def main():
    # Header
    st.title("🇹🇷 TRY Carry Trade Analysis Dashboard")
    st.markdown(
        '<div class="baseline-label">Baseline = Convert now at bank rate (43.10) + T-Bill @ 3.5%</div>',
        unsafe_allow_html=True
    )

    # Load data
    with st.spinner("Loading market data..."):
        df = load_market_data()
        spot_series = get_spot_series(df)
        data_summary = get_data_summary(df)

    # Sidebar - Parameters
    with st.sidebar:
        st.header("📝 Trade Parameters")

        st.subheader("Principal & Rates")
        principal_try = st.number_input(
            "Principal (TRY)",
            value=1_430_000,
            min_value=10_000,
            max_value=100_000_000,
            step=10_000,
            format="%d"
        )

        deposit_rate = st.slider(
            "Deposit Rate (% annual)",
            min_value=10.0,
            max_value=60.0,
            value=39.5,
            step=0.5
        )

        term_days = st.number_input(
            "Term (calendar days)",
            value=32,
            min_value=7,
            max_value=365,
            step=1
        )

        stopaj_rate = st.slider(
            "Stopaj Tax (%)",
            min_value=0.0,
            max_value=30.0,
            value=17.5,
            step=0.5
        )

        st.subheader("Exchange Rates")

        # Get spot at entry date
        entry_date_spot, entry_spot_value = get_spot_at_date(spot_series, "2025-12-17")
        default_spot = entry_spot_value if entry_spot_value else 42.72

        spot_entry = st.number_input(
            "Spot at Entry (USD/TRY)",
            value=float(default_spot),
            min_value=30.0,
            max_value=60.0,
            step=0.01,
            format="%.4f"
        )

        entry_bank_rate = st.number_input(
            "Entry Bank Rate (TRY/USD)",
            value=43.10,
            min_value=30.0,
            max_value=65.0,
            step=0.01,
            format="%.4f",
            help="Executable rate for TRY→USD at entry"
        )

        swift_fee = st.slider(
            "SWIFT Fee ($)",
            min_value=0.0,
            max_value=100.0,
            value=32.50,
            step=2.5
        )

        st.subheader("Risk-Free Rate")
        usd_rf_rate = st.slider(
            "US T-Bill Rate (% annual)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.25
        )

        st.subheader("Regime Selection")
        regime_options = get_regime_options()
        regime_name = st.selectbox(
            "Calibration Window",
            options=list(regime_options.keys()),
            index=1,  # Default to 2 Years
            help="Time window for volatility calibration and backtest"
        )
        regime_days = regime_options[regime_name]

        st.subheader("Monte Carlo")
        n_sims = st.select_slider(
            "Simulations",
            options=[10000, 25000, 50000, 100000],
            value=50000
        )

        # Final TRY override
        st.subheader("Optional Overrides")
        use_final_override = st.checkbox("Override Final TRY", value=False)
        if use_final_override:
            final_try_override = st.number_input(
                "Final TRY",
                value=1_470_854.91,
                min_value=principal_try,
                step=100.0,
                format="%.2f"
            )
        else:
            final_try_override = None

    # Build params
    params = TradeParams(
        principal_try=principal_try,
        deposit_rate_annual=deposit_rate / 100,
        term_days_calendar=term_days,
        stopaj_rate=stopaj_rate / 100,
        entry_bank_rate=entry_bank_rate,
        spot_entry=spot_entry,
        swift_fee_usd=swift_fee,
        usd_rf_rate_annual=usd_rf_rate / 100,
    )

    # Override final TRY if specified
    if final_try_override:
        params.final_try = final_try_override
        # Recalculate break-even
        params.bank_rate_be = params.final_try / (params.usd_rf_end + params.swift_fee_usd)
        params.spot_be = params.bank_rate_be / (1 + params.spread)
        params.be_move_pct = (params.spot_be / params.spot_entry - 1) * 100

    model = CarryTradeModel(params)

    # Validate data sufficiency
    validation = validate_data_sufficiency(spot_series, regime_days)
    if not validation['sufficient']:
        st.warning(f"⚠️ Insufficient data for {regime_name}. Using fallback: {validation['fallback_regime']} days")
        regime_days = validation['fallback_regime']

    # Calibrate GBM
    calib = calibrate_gbm(spot_series, regime_days)

    # Run Monte Carlo
    mc_results = model.monte_carlo(
        mu_annual=calib['mu_annual'],
        sigma_annual=calib['sigma_annual'],
        n_sims=n_sims
    )

    # Run backtest
    backtest = model.historical_backtest(spot_series, regime_days)

    # Scenarios
    scenarios = model.scenario_analysis()

    # =========================================
    # MAIN CONTENT
    # =========================================

    # KPI Cards
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "USD Invested",
            format_currency(params.usd0_exec),
            help="Executable USD at entry bank rate"
        )
    with col2:
        st.metric(
            "T-Bill End Value",
            format_currency(params.usd_rf_end),
            f"{usd_rf_rate}% for {term_days}d"
        )
    with col3:
        st.metric(
            "Break-even Spot",
            format_number(params.spot_be),
            format_percent(params.be_move_pct)
        )
    with col4:
        st.metric(
            "Bank Spread",
            format_percent(params.spread * 100, with_sign=False),
            "Derived from entry rates"
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        delta_color = "normal" if mc_results['mean_excess'] >= 0 else "inverse"
        st.metric(
            "Expected Excess",
            format_percent(mc_results['mean_excess']),
            "vs T-Bill",
            delta_color=delta_color
        )
    with col6:
        st.metric(
            "Sharpe Ratio",
            f"{mc_results['sharpe']:.3f}",
            "Period excess / std"
        )
    with col7:
        st.metric(
            "P(Underperform)",
            f"{mc_results['prob_underperform_tbill']:.1f}%",
            "vs T-Bill"
        )
    with col8:
        st.metric(
            "VaR 95%",
            format_percent(mc_results['var_95_excess']),
            "Excess return"
        )

    # Verdict
    st.markdown(create_verdict_html(mc_results, params.__dict__), unsafe_allow_html=True)

    # Compute real-time analysis
    pnl_realtime = compute_realtime_pnl(params, spot_series, entry_date="2025-12-19")
    trend_analysis = compute_trend_analysis(spot_series, pnl_realtime.days_remaining, pnl_realtime.current_spot)
    cushion_analysis = compute_daily_cushion(params, pnl_realtime.current_spot, pnl_realtime.days_remaining, trend_analysis)
    adjusted_returns = compute_adjusted_expected_return(params, trend_analysis, mc_results, pnl_realtime.days_remaining)

    # Tabs for different analyses
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⚡ Real-Time",
        "📈 Scenarios",
        "📉 Payoff",
        "🎲 Monte Carlo",
        "📜 Backtest",
        "🌍 Regimes",
        "🔧 Debug"
    ])

    with tab0:
        st.subheader("Real-Time Position Analysis")

        # Status banner
        spot_move = (pnl_realtime.current_spot / pnl_realtime.entry_spot - 1) * 100
        status_color = "🟢" if pnl_realtime.unrealized_pnl_usd >= 0 else "🔴"
        st.info(f"{status_color} **Entry:** 12/19/2025 @ {pnl_realtime.entry_spot:.4f} | **Now:** {pnl_realtime.current_date.strftime('%m/%d/%Y')} @ {pnl_realtime.current_spot:.4f} ({spot_move:+.2f}%) | **{pnl_realtime.days_elapsed}d elapsed, {pnl_realtime.days_remaining}d remaining**")

        # ROW 1: Key P&L Metrics
        st.markdown("### Position P&L")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        with col_p1:
            st.metric(
                "Unrealized P&L",
                f"${pnl_realtime.unrealized_pnl_usd:+,.2f}",
                f"{pnl_realtime.unrealized_return_pct:+.2f}%"
            )

        with col_p2:
            st.metric(
                "vs T-Bill",
                f"${pnl_realtime.excess_vs_tbill_now:+,.2f}",
                "Excess return"
            )

        with col_p3:
            st.metric(
                "MTM Return",
                f"{pnl_realtime.mtm_return_pct:+.2f}%",
                "At current spot"
            )

        with col_p4:
            st.metric(
                "USD Invested",
                f"${pnl_realtime.usd0_exec:,.2f}",
                f"@ {pnl_realtime.entry_bank_rate:.4f}"
            )

        # ROW 2: Cushion & Trend Metrics
        st.markdown("### Daily Cushion Analysis")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)

        with col_c1:
            st.metric(
                "Total Cushion",
                f"{cushion_analysis.cushion_pct:.2f}%",
                f"BE @ {cushion_analysis.spot_be:.4f}"
            )

        with col_c2:
            st.metric(
                "Daily Cushion",
                f"{cushion_analysis.daily_cushion_pct:.3f}%/day",
                f"{cushion_analysis.daily_cushion_absolute:.4f} TRY"
            )

        with col_c3:
            st.metric(
                "Daily Trend",
                f"{trend_analysis.slope_pct_daily:.3f}%/day",
                f"{trend_analysis.accel_regime}"
            )

        with col_c4:
            ratio = cushion_analysis.cushion_vs_trend_ratio
            days_be = cushion_analysis.days_until_breakeven
            days_be_str = f"{days_be:.0f}d" if days_be < 1000 else "∞"
            st.metric(
                "Cushion/Trend",
                f"{ratio:.1f}x",
                f"BE in {days_be_str}"
            )

        # Status box
        status = cushion_analysis.status
        if "DANGER" in status:
            st.error(f"⚠️ **{status}**")
        elif "CAUTION" in status or "WARNING" in status:
            st.warning(f"⚡ **{status}**")
        else:
            st.success(f"✅ **{status}**")

        # Prepare data dicts for charts
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
            'spot_be': params.spot_be,
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

        # Charts
        st.markdown("### Charts")

        # Chart row 1
        col_rt1, col_rt2 = st.columns(2)

        with col_rt1:
            try:
                fig_pnl = create_realtime_pnl_chart(pnl_dict, spot_series)
                st.plotly_chart(fig_pnl, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        with col_rt2:
            try:
                fig_trend = create_trend_chart(spot_series, trend_dict, pnl_realtime.days_remaining)
                st.plotly_chart(fig_trend, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        # Chart row 2
        col_rt3, col_rt4 = st.columns(2)

        with col_rt3:
            try:
                fig_gauge = create_cushion_gauge(cushion_dict)
                st.plotly_chart(fig_gauge, use_container_width=True)
            except Exception as e:
                st.error(f"Gauge error: {e}")

        with col_rt4:
            try:
                fig_timeline = create_cushion_timeline(cushion_dict, trend_dict)
                st.plotly_chart(fig_timeline, use_container_width=True)
            except Exception as e:
                st.error(f"Timeline error: {e}")

        # Trend-adjusted returns
        st.markdown("---")
        st.markdown("### Trend-Adjusted Expected Return")

        col_adj1, col_adj2, col_adj3 = st.columns(3)

        with col_adj1:
            st.metric(
                "Base Expected",
                f"{adjusted_returns['base_expected_excess']:+.2f}%",
                "From MC"
            )

        with col_adj2:
            st.metric(
                "Trend-Adjusted",
                f"{adjusted_returns['adjusted_expected_excess']:+.2f}%",
                f"{adjusted_returns['adjustment_factor']:.2f}x factor"
            )

        with col_adj3:
            st.metric(
                "P(Underperform)",
                f"{adjusted_returns['adjusted_prob_underperform']:.1f}%",
                f"Base: {adjusted_returns['base_prob_underperform']:.1f}%"
            )

        # Detailed data expanders
        with st.expander("📊 Entry & Current Details"):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Entry (12/19/2025)**")
                st.text(f"Spot:        {pnl_realtime.entry_spot:.4f}")
                st.text(f"Bank Rate:   {pnl_realtime.entry_bank_rate:.4f}")
                st.text(f"USD Invested: ${pnl_realtime.usd0_exec:,.2f}")

            with col_d2:
                st.markdown("**Current**")
                st.text(f"Spot:        {pnl_realtime.current_spot:.4f}")
                st.text(f"Bank Rate:   {pnl_realtime.current_bank_rate:.4f}")
                st.text(f"TRY Value:   {pnl_realtime.current_try_value:,.2f}")
                st.text(f"USD Value:   ${pnl_realtime.current_usd_value:,.2f}")

        with st.expander("📈 Trend Details"):
            st.text(f"1-Week Slope:    {trend_analysis.slope_1w:.4f} TRY/day")
            st.text(f"2-Week Slope:    {trend_analysis.slope_2w:.4f} TRY/day")
            st.text(f"1-Month Slope:   {trend_analysis.slope_1m:.4f} TRY/day")
            st.text(f"20-Day Slope:    {trend_analysis.slope_daily:.4f} TRY/day ({trend_analysis.slope_pct_daily:.3f}%)")
            st.text(f"Annualized:      {trend_analysis.slope_annual:.2f} TRY/year ({trend_analysis.slope_annual/pnl_realtime.current_spot*100:.1f}%)")
            st.text(f"R-squared:       {trend_analysis.r_squared:.3f}")
            st.text(f"Acceleration:    {trend_analysis.accel_daily:.4f} ({trend_analysis.accel_regime})")
            st.text(f"Projected Spot:  {trend_analysis.projected_spot_maturity:.4f} ({trend_analysis.projected_move_pct:+.2f}%)")

        with st.expander("📄 Full Report"):
            report = generate_realtime_report(pnl_realtime, trend_analysis, cushion_analysis, adjusted_returns)
            st.code(report)

    with tab1:
        st.subheader("Scenario Analysis")
        st.markdown("P/L for different USD/TRY movements at maturity")

        fig_scenario = create_scenario_chart(scenarios, params.__dict__)
        st.plotly_chart(fig_scenario, use_container_width=True)

        # Scenario table
        with st.expander("📋 Scenario Table (click to expand)"):
            display_df = scenarios[[
                'spot_move_pct', 'spot_end', 'bank_rate_end', 'usd_end',
                'pnl_vs_convert_now', 'pnl_vs_tbill', 'ret_vs_convert_now', 'excess_ret_vs_tbill'
            ]].copy()
            display_df.columns = [
                'Move %', 'Spot', 'Bank Rate', 'USD End',
                'P/L vs Convert ($)', 'P/L vs T-Bill ($)', 'Ret vs Convert (%)', 'Excess vs T-Bill (%)'
            ]
            st.dataframe(display_df.style.format({
                'Move %': '{:+.1f}',
                'Spot': '{:.4f}',
                'Bank Rate': '{:.4f}',
                'USD End': '${:,.2f}',
                'P/L vs Convert ($)': '${:+,.2f}',
                'P/L vs T-Bill ($)': '${:+,.2f}',
                'Ret vs Convert (%)': '{:+.2f}',
                'Excess vs T-Bill (%)': '{:+.2f}',
            }), use_container_width=True)

    with tab2:
        st.subheader("Payoff Diagram")
        st.markdown("Return profile across USD/TRY range")

        fig_payoff = create_payoff_diagram(params.__dict__)
        st.plotly_chart(fig_payoff, use_container_width=True)

    with tab3:
        st.subheader("Monte Carlo Simulation")
        st.markdown(f"**{n_sims:,} simulations** | μ = {calib['mu_annual']*100:.1f}% | σ = {calib['sigma_annual']*100:.1f}%")

        col_mc1, col_mc2 = st.columns(2)

        with col_mc1:
            st.markdown("#### Returns vs Convert Now")
            fig_mc1 = create_mc_distribution(mc_results, show_excess=False)
            st.plotly_chart(fig_mc1, use_container_width=True)

        with col_mc2:
            st.markdown("#### Excess Returns vs T-Bill")
            fig_mc2 = create_mc_distribution(mc_results, show_excess=True)
            st.plotly_chart(fig_mc2, use_container_width=True)

        # MC Statistics
        with st.expander("📊 Monte Carlo Statistics"):
            mc_stats = pd.DataFrame({
                'Metric': [
                    'Mean Return', 'Median Return', 'Std Dev', 'P5', 'P95',
                    'VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%',
                    'P(Loss vs Convert)', 'P(Underperform T-Bill)', 'Sharpe'
                ],
                'vs Convert Now': [
                    f"{mc_results['mean_ret']:.2f}%",
                    f"{mc_results['median_ret']:.2f}%",
                    f"{mc_results['std_ret']:.2f}%",
                    f"{mc_results['p5_ret']:.2f}%",
                    f"{mc_results['p95_ret']:.2f}%",
                    f"{mc_results['var_95_ret']:.2f}%",
                    f"{mc_results['var_99_ret']:.2f}%",
                    f"{mc_results['cvar_95_ret']:.2f}%",
                    f"{mc_results['cvar_99_ret']:.2f}%",
                    f"{mc_results['prob_loss_vs_convert']:.1f}%",
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
            st.markdown(f"**{backtest['n_windows']} rolling windows** from {regime_name} data")

            fig_bt = create_backtest_chart(backtest)
            st.plotly_chart(fig_bt, use_container_width=True)

            # Stats
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
        st.subheader("Regime Comparison")
        st.markdown("Compare results across different calibration windows")

        with st.spinner("Running regime comparison..."):
            regime_df = run_regime_comparison(spot_series, params, n_sims=min(n_sims, 25000))

        fig_regime = create_regime_comparison_chart(regime_df)
        st.plotly_chart(fig_regime, use_container_width=True)

        # Regime comparison table
        st.markdown("#### Summary Table")
        display_cols = [
            'regime', 'n_returns', 'mu_annual', 'sigma_annual',
            'mc_mean_excess', 'mc_prob_underperform', 'mc_sharpe',
            'bt_win_rate'
        ]
        display_df = regime_df[display_cols].copy()
        display_df.columns = [
            'Regime', 'N Returns', 'μ (% ann)', 'σ (% ann)',
            'Mean Excess (%)', 'P(Under) %', 'Sharpe', 'BT Win Rate %'
        ]
        st.dataframe(display_df.style.format({
            'N Returns': '{:,.0f}',
            'μ (% ann)': '{:.1f}',
            'σ (% ann)': '{:.1f}',
            'Mean Excess (%)': '{:+.2f}',
            'P(Under) %': '{:.1f}',
            'Sharpe': '{:.3f}',
            'BT Win Rate %': '{:.1f}',
        }), use_container_width=True)

    with tab6:
        st.subheader("Debug & Verification")

        # Consistency check
        checks = run_consistency_check(params)

        st.markdown("#### Consistency Check Results")

        if checks['verification']['ALL_CHECKS_PASS']:
            st.success("✅ All consistency checks PASSED")
        else:
            st.error("❌ Some checks FAILED - review model")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("**Computed Values:**")
            for k, v in checks['computed'].items():
                if isinstance(v, float):
                    st.text(f"{k}: {v:,.4f}")
                else:
                    st.text(f"{k}: {v}")

        with col_d2:
            st.markdown("**Verification:**")
            for k, v in checks['verification'].items():
                if isinstance(v, bool):
                    status = "✅ PASS" if v else "❌ FAIL"
                    st.text(f"{k}: {status}")
                elif isinstance(v, float):
                    st.text(f"{k}: {v:,.4f}")

        # Trade economics summary
        with st.expander("📄 Full Trade Economics"):
            summary_text = compute_trade_economics_summary(params.__dict__)
            st.code(summary_text)

        # Data info
        with st.expander("📊 Data Summary"):
            for k, v in data_summary.items():
                st.text(f"{k}: {v}")

    # Historical chart (always visible at bottom)
    st.markdown("---")
    st.subheader("📈 USD/TRY Historical")
    fig_hist = create_historical_chart(spot_series.tail(365), params.__dict__)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Export section
    st.markdown("---")
    st.subheader("📥 Export")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        # Scenario CSV
        csv_scenarios = generate_scenario_table_csv(scenarios)
        st.download_button(
            "📋 Download Scenarios (CSV)",
            csv_scenarios,
            "scenarios.csv",
            "text/csv"
        )

    with col_exp2:
        # Metrics CSV
        csv_metrics = generate_metrics_csv(params.__dict__, mc_results, backtest)
        st.download_button(
            "📊 Download Metrics (CSV)",
            csv_metrics,
            "metrics.csv",
            "text/csv"
        )

    with col_exp3:
        # Full report text
        report_text = compute_trade_economics_summary(params.__dict__)
        report_text += "\n\n" + print_consistency_report(checks)
        st.download_button(
            "📄 Download Full Report (TXT)",
            report_text,
            "carry_trade_report.txt",
            "text/plain"
        )

    # Footer
    st.markdown("---")
    st.caption(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data: {data_summary['start_date']} to {data_summary['end_date']} | "
        f"Model: Opportunity-Cost Baseline"
    )


if __name__ == "__main__":
    main()
