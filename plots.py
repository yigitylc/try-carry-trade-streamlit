"""
plots.py
========
Premium Plotly visualizations for TRY Carry Trade Dashboard.

Features:
- Dark theme with consistent styling
- Annotated interactive charts
- KPI cards and gauges
- Export to HTML
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Premium dark theme colors
COLORS = {
    'bg_dark': '#0e1117',
    'bg_card': '#1a1d24',
    'bg_hover': '#262b36',
    'text_primary': '#fafafa',
    'text_secondary': '#a0a0a0',
    'text_muted': '#6c757d',
    'profit': '#00d26a',
    'profit_light': '#00ff88',
    'loss': '#ff4757',
    'loss_light': '#ff6b7a',
    'warning': '#ffa502',
    'info': '#3498db',
    'tbill': '#9b59b6',
    'neutral': '#74b9ff',
    'grid': '#2d3436',
    'breakeven': '#f39c12',
}

# Chart layout defaults
LAYOUT_DEFAULTS = {
    'paper_bgcolor': COLORS['bg_dark'],
    'plot_bgcolor': COLORS['bg_card'],
    'font': {'family': 'Inter, Arial, sans-serif', 'color': COLORS['text_primary']},
    'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60},
    'hovermode': 'x unified',
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme to a Plotly figure."""
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        zerolinecolor=COLORS['grid'],
        tickfont={'color': COLORS['text_secondary']},
    )
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        zerolinecolor=COLORS['grid'],
        tickfont={'color': COLORS['text_secondary']},
    )
    return fig


def create_scenario_chart(scenarios_df: pd.DataFrame, params: dict) -> go.Figure:
    """
    Create scenario analysis bar chart.

    Shows P/L vs Baseline and Excess vs T-Bill for each spot movement.
    Shows P/L vs Baseline and Excess vs T-Bill for each spot movement.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            'P/L vs Baseline at Entry ($)',
            'P/L vs Baseline at Entry ($)',
            'Excess Return vs T-Bill (%)'
        )
    )

    # Colors based on profit/loss
    colors_pnl = [COLORS['profit'] if x >= 0 else COLORS['loss']
                  for x in scenarios_df['pnl_vs_baseline']]
    colors_pnl = [COLORS['profit'] if x >= 0 else COLORS['loss']
                  for x in scenarios_df['pnl_vs_baseline']]
    colors_excess = [COLORS['profit'] if x >= 0 else COLORS['loss']
                     for x in scenarios_df['excess_ret_vs_tbill']]

    # P/L bars
    fig.add_trace(
        go.Bar(
            x=scenarios_df['spot_move_pct'],
            y=scenarios_df['pnl_vs_baseline'],
            marker_color=colors_pnl,
            name='P/L vs Baseline',
            text=[f"${x:+,.0f}" for x in scenarios_df['pnl_vs_baseline']],
            textposition='outside',
            hovertemplate='Spot Move: %{x:.1f}%<br>P/L: $%{y:,.2f}<extra></extra>',
        ),
            y=scenarios_df['pnl_vs_baseline'],
            marker_color=colors_pnl,
            name='P/L vs Baseline',
            text=[f"${x:+,.0f}" for x in scenarios_df['pnl_vs_baseline']],
            textposition='outside',
            hovertemplate='Spot Move: %{x:.1f}%<br>P/L: $%{y:,.2f}<extra></extra>',
        ),
        row=1, col=1
    )

    # Excess return bars
    fig.add_trace(
        go.Bar(
            x=scenarios_df['spot_move_pct'],
            y=scenarios_df['excess_ret_vs_tbill'],
            marker_color=colors_excess,
            name='Excess vs T-Bill',
            text=[f"{x:+.2f}%" for x in scenarios_df['excess_ret_vs_tbill']],
            textposition='outside',
            hovertemplate='Spot Move: %{x:.1f}%<br>Excess: %{y:.2f}%<extra></extra>',
        ),
        row=2, col=1
    )

    # Break-even line
    be_move = params.get('be_move_pct', 0)
    for row in [1, 2]:
        fig.add_vline(
            x=be_move, line_dash="dash", line_color=COLORS['breakeven'],
            annotation_text=f"BE: {be_move:.1f}%",
            annotation_position="top",
            row=row, col=1
        )

    # Zero lines
    fig.add_hline(y=0, line_color=COLORS['text_muted'], line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_color=COLORS['text_muted'], line_width=1, row=2, col=1)

    fig.update_layout(
        title={
            'text': 'Scenario Analysis: Carry Trade Outcomes by USD/TRY Movement',
            'font': {'size': 16, 'color': COLORS['text_primary']},
        },
        showlegend=False,
        height=600,
    )

    fig.update_xaxes(title_text="USD/TRY Spot Movement (%)", row=2, col=1)
    fig.update_yaxes(title_text="P/L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Excess Return (%)", row=2, col=1)

    return apply_dark_theme(fig)


def create_payoff_diagram(params: dict, spot_range: tuple = None) -> go.Figure:
    """
    Create payoff diagram showing USD outcome vs spot rate.

    Includes:
    - Carry trade payoff curve
    - T-Bill reference line
    - Break-even annotation
    - Entry spot marker
    """
    if spot_range is None:
        spot_entry = params['spot_entry']
        spot_range = (spot_entry * 0.85, spot_entry * 1.25)

    spots = np.linspace(spot_range[0], spot_range[1], 200)

    # Calculate USD outcomes
    exit_spread = params['exit_spread']
    final_try = params['final_try']
    swift_fee = params['swift_fee_usd']
    usd0_exec = params['usd0_baseline']
    usd_rf_end = params['usd_rf_end']
    spot_be = params['spot_be']

    usd_ends = []
    for spot in spots:
        bank_rate = spot * (1 + exit_spread)
        usd_end = final_try / bank_rate - swift_fee
        usd_ends.append(usd_end)
    exit_spread = params['exit_spread']
    final_try = params['final_try']
    swift_fee = params['swift_fee_usd']
    usd0_exec = params['usd0_baseline']
    usd_rf_end = params['usd_rf_end']
    spot_be = params['spot_be']

    usd_ends = []
    for spot in spots:
        bank_rate = spot * (1 + exit_spread)
        usd_end = final_try / bank_rate - swift_fee
        usd_ends.append(usd_end)

    usd_ends = np.array(usd_ends)
    ret_vs_convert = (usd_ends / usd0_exec - 1) * 100

    fig = go.Figure()

    # Fill areas
    fig.add_trace(go.Scatter(
        x=spots[ret_vs_convert >= 0],
        y=ret_vs_convert[ret_vs_convert >= 0],
        fill='tozeroy',
        fillcolor=f'rgba(0, 210, 106, 0.2)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.add_trace(go.Scatter(
        x=spots[ret_vs_convert < 0],
        y=ret_vs_convert[ret_vs_convert < 0],
        fill='tozeroy',
        fillcolor=f'rgba(255, 71, 87, 0.2)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Main payoff line
    fig.add_trace(go.Scatter(
        x=spots,
        y=ret_vs_convert,
        mode='lines',
        name='Carry Trade Return',
        line=dict(color=COLORS['neutral'], width=3),
        hovertemplate='Spot: %{x:.2f}<br>Return: %{y:.2f}%<extra></extra>',
    ))

    # T-Bill line
    rf_return = (usd_rf_end / usd0_exec - 1) * 100
    fig.add_hline(y=rf_return, line_dash="dash", line_color=COLORS['tbill'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=rf_return,
        text=f"T-Bill: {rf_return:.2f}%",
        showarrow=False,
        font=dict(color=COLORS['tbill']),
    )

    # Break-even line
    fig.add_vline(x=spot_be, line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=spot_be,
        text=f"Break-even: {spot_be:.2f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )

    # Entry spot
    fig.add_vline(x=params['spot_entry'], line_dash="dot", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.02,
        align='left',
        y=params['spot_entry'],
        text=f"Entry: {params['spot_entry']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )
    fig.add_hline(y=rf_return, line_dash="dash", line_color=COLORS['tbill'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=rf_return,
        text=f"T-Bill: {rf_return:.2f}%",
        showarrow=False,
        font=dict(color=COLORS['tbill']),
    )

    # Break-even line
    fig.add_vline(x=spot_be, line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=spot_be,
        text=f"Break-even: {spot_be:.2f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )

    # Entry spot
    fig.add_vline(x=params['spot_entry'], line_dash="dot", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.02,
        align='left',
        y=params['spot_entry'],
        text=f"Entry: {params['spot_entry']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )

    # Zero line
    fig.add_hline(y=0, line_color=COLORS['text_muted'], line_width=1)

    x_min = min(spot_range[0], params['spot_entry'], spot_be)
    x_max = max(spot_range[1], params['spot_entry'], spot_be)

    fig.update_layout(
        title={
            'text': 'Payoff Diagram: Return vs USD/TRY at Maturity',
            'font': {'size': 16},
        },
        xaxis_title='USD/TRY Spot at Maturity',
        xaxis_range=[x_min, x_max],
        yaxis_title='Return vs Baseline (%)',
        height=500,
    x_min = min(spot_range[0], params['spot_entry'], spot_be)
    x_max = max(spot_range[1], params['spot_entry'], spot_be)

    fig.update_layout(
        title={
            'text': 'Payoff Diagram: Return vs USD/TRY at Maturity',
            'font': {'size': 16},
        },
        xaxis_title='USD/TRY Spot at Maturity',
        xaxis_range=[x_min, x_max],
        yaxis_title='Return vs Baseline (%)',
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )

    return apply_dark_theme(fig)


def create_mc_distribution(mc_results: dict, show_excess: bool = True) -> go.Figure:
    """
    Create Monte Carlo distribution chart.

    Shows histogram of returns with VaR/CVaR markers.
    """
    if show_excess:
        data = mc_results['excess_ret']
        title = 'Monte Carlo: Excess Return vs T-Bill Distribution'
        xlabel = 'Excess Return vs T-Bill (%)'
        mean_val = mc_results['mean_excess']
        var_95 = mc_results['var_95_excess']
        cvar_95 = mc_results['cvar_95_excess']
        prob_loss = mc_results['prob_underperform_tbill']
        loss_label = 'Underperform T-Bill'
    else:
        data = mc_results['ret_vs_baseline']
        title = 'Monte Carlo: Return vs Baseline Distribution'
        xlabel = 'Return vs Baseline (%)'
        mean_val = mc_results['mean_ret']
        var_95 = mc_results['var_95_ret']
        cvar_95 = mc_results['cvar_95_ret']
        prob_loss = mc_results['prob_loss_vs_baseline']
        loss_label = 'Loss'
    else:
        data = mc_results['ret_vs_baseline']
        title = 'Monte Carlo: Return vs Baseline Distribution'
        xlabel = 'Return vs Baseline (%)'
        mean_val = mc_results['mean_ret']
        var_95 = mc_results['var_95_ret']
        cvar_95 = mc_results['cvar_95_ret']
        prob_loss = mc_results['prob_loss_vs_baseline']
        loss_label = 'Loss'

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=100,
        marker_color=COLORS['neutral'],
        opacity=0.7,
        name='Distribution',
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>',
    ))

    # Zero line (break-even)
    fig.add_vline(x=0, line_color=COLORS['text_primary'], line_width=2)
    fig.add_annotation(x=0, y=1, yref='paper', text="BE", showarrow=False,
                       font=dict(color=COLORS['text_primary'], size=11), yshift=10)

    # Mean line
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS['info'], line_width=2)
    fig.add_annotation(x=mean_val, y=0.9, yref='paper', text=f"Mean<br>{mean_val:.2f}%",
                       showarrow=True, arrowhead=2, arrowcolor=COLORS['info'],
                       font=dict(color=COLORS['info'], size=10), bgcolor=COLORS['bg_card'],
                       bordercolor=COLORS['info'], borderwidth=1)

    # VaR 95% line
    fig.add_vline(x=var_95, line_dash="dot", line_color=COLORS['warning'], line_width=2)
    fig.add_annotation(x=var_95, y=0.7, yref='paper', text=f"VaR 95%<br>{var_95:.2f}%",
                       showarrow=True, arrowhead=2, arrowcolor=COLORS['warning'],
                       font=dict(color=COLORS['warning'], size=10), bgcolor=COLORS['bg_card'],
                       bordercolor=COLORS['warning'], borderwidth=1)

    # CVaR 95% line
    fig.add_vline(x=cvar_95, line_dash="dot", line_color=COLORS['loss'], line_width=2)
    fig.add_annotation(x=cvar_95, y=0.5, yref='paper', text=f"CVaR 95%<br>{cvar_95:.2f}%",
                       showarrow=True, arrowhead=2, arrowcolor=COLORS['loss'],
                       font=dict(color=COLORS['loss'], size=10), bgcolor=COLORS['bg_card'],
                       bordercolor=COLORS['loss'], borderwidth=1)

    fig.update_layout(
        title={
            'text': f'{title}<br><sub>P({loss_label}) = {prob_loss:.1f}% | Sharpe = {mc_results["sharpe"]:.3f}</sub>',
            'font': {'size': 14},
        },
        xaxis_title=xlabel,
        yaxis_title='Frequency',
        height=450,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def create_historical_chart(spot_series: pd.Series, params: dict) -> go.Figure:
    """
    Create historical USD/TRY chart with trade levels.
    """
    fig = go.Figure()

    # Main price line
    fig.add_trace(go.Scatter(
        x=spot_series.index,
        y=spot_series.values,
        mode='lines',
        name='USD/TRY',
        line=dict(color=COLORS['neutral'], width=2),
        hovertemplate='Date: %{x}<br>Spot: %{y:.4f}<extra></extra>',
    ))

    # Entry spot line
    fig.add_hline(y=params['spot_entry'], line_dash="dash", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=params['spot_entry'],
        text=f"Entry Spot: {params['spot_entry']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )

    # Break-even line
    fig.add_hline(y=params['spot_be'], line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=params['spot_be'],
        text=f"Break-even: {params['spot_be']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )

    # Entry bank rate line
    fig.add_hline(y=params['entry_bank_rate'], line_dash="dot", line_color=COLORS['tbill'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=params['entry_bank_rate'],
        text=f"Entry Bank Rate: {params['entry_bank_rate']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['tbill']),
    )

    # Calculate Y-axis range to include entry, break-even, and bank rate
    y_min = min(spot_series.min(), params['spot_entry'], params['spot_be'], params['entry_bank_rate']) * 0.98
    y_max = max(spot_series.max(), params['spot_entry'], params['spot_be'], params['entry_bank_rate']) * 1.02
    fig.add_hline(y=params['spot_entry'], line_dash="dash", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=params['spot_entry'],
        text=f"Entry Spot: {params['spot_entry']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )

    # Break-even line
    fig.add_hline(y=params['spot_be'], line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=params['spot_be'],
        text=f"Break-even: {params['spot_be']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )

    # Entry bank rate line
    fig.add_hline(y=params['entry_bank_rate'], line_dash="dot", line_color=COLORS['tbill'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=params['entry_bank_rate'],
        text=f"Entry Bank Rate: {params['entry_bank_rate']:.2f}",
        showarrow=False,
        font=dict(color=COLORS['tbill']),
    )

    # Calculate Y-axis range to include entry, break-even, and bank rate
    y_min = min(spot_series.min(), params['spot_entry'], params['spot_be'], params['entry_bank_rate']) * 0.98
    y_max = max(spot_series.max(), params['spot_entry'], params['spot_be'], params['entry_bank_rate']) * 1.02

    fig.update_layout(
        title={
            'text': 'USD/TRY Historical with Trade Levels',
            'font': {'size': 16},
        },
        xaxis_title='Date',
        yaxis_title='USD/TRY',
        yaxis_range=[y_min, y_max],
        height=400,
        xaxis_rangeslider_visible=True,
    )

    return apply_dark_theme(fig)


def create_backtest_chart(backtest_results: dict) -> go.Figure:
    """
    Create historical backtest distribution chart.
    """
    if 'error' in backtest_results or backtest_results.get('n_windows', 0) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for backtest",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['warning'])
        )
        return apply_dark_theme(fig)

    data = backtest_results['data']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Return vs Baseline (%)',
            'Return vs Baseline (%)',
            'Excess Return vs T-Bill (%)'
        )
    )

    # Return vs Convert histogram
    fig.add_trace(
        go.Histogram(
            x=data['ret_vs_baseline'],
            x=data['ret_vs_baseline'],
            nbinsx=50,
            marker_color=COLORS['neutral'],
            opacity=0.7,
            name='Return',
        ),
        row=1, col=1
    )

    # Excess return histogram
    fig.add_trace(
        go.Histogram(
            x=data['excess_ret'],
            nbinsx=50,
            marker_color=COLORS['tbill'],
            opacity=0.7,
            name='Excess',
        ),
        row=1, col=2
    )

    # Zero lines
    fig.add_vline(x=0, line_color=COLORS['text_primary'], line_width=1, row=1, col=1)
    fig.add_vline(x=0, line_color=COLORS['text_primary'], line_width=1, row=1, col=2)

    # Mean lines
    fig.add_vline(x=backtest_results['mean_ret'], line_dash="dash",
                  line_color=COLORS['info'], row=1, col=1)
    fig.add_vline(x=backtest_results['mean_excess'], line_dash="dash",
                  line_color=COLORS['info'], row=1, col=2)

    win_rate = backtest_results['win_rate_vs_tbill']
    n_windows = backtest_results['n_windows']

    fig.update_layout(
        title={
            'text': f'Historical Backtest ({n_windows} windows)<br><sub>Win Rate vs T-Bill: {win_rate:.1f}%</sub>',
            'font': {'size': 14},
        },
        height=400,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def create_regime_comparison_chart(regime_df: pd.DataFrame) -> go.Figure:
    """
    Create regime comparison bar chart.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Mean Excess Return (%)',
            'P(Underperform T-Bill) %',
            'VaR 95% Excess (%)',
            'Sharpe Ratio'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    regimes = regime_df['regime'].tolist()

    # Mean excess
    colors = [COLORS['profit'] if x >= 0 else COLORS['loss']
              for x in regime_df['mc_mean_excess']]
    fig.add_trace(
        go.Bar(x=regimes, y=regime_df['mc_mean_excess'], marker_color=colors, name='Mean'),
        row=1, col=1
    )

    # Prob underperform
    fig.add_trace(
        go.Bar(x=regimes, y=regime_df['mc_prob_underperform'],
               marker_color=COLORS['loss'], name='P(Under)'),
        row=1, col=2
    )

    # VaR 95%
    fig.add_trace(
        go.Bar(x=regimes, y=regime_df['mc_var95_excess'],
               marker_color=COLORS['warning'], name='VaR'),
        row=2, col=1
    )

    # Sharpe
    colors_sharpe = [COLORS['profit'] if x >= 0 else COLORS['loss']
                     for x in regime_df['mc_sharpe']]
    fig.add_trace(
        go.Bar(x=regimes, y=regime_df['mc_sharpe'], marker_color=colors_sharpe, name='Sharpe'),
        row=2, col=2
    )

    fig.update_layout(
        title={
            'text': 'Regime Comparison: Monte Carlo Results Across Time Windows',
            'font': {'size': 16},
        },
        height=600,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def create_kpi_cards_html(params: dict, mc_results: dict = None) -> str:
    """
    Generate HTML for KPI cards.
    """
    usd0 = params['usd0_baseline']
    usd0 = params['usd0_baseline']
    usd_rf = params['usd_rf_end']
    spot_be = params['spot_be']
    be_move = params['be_move_pct']

    cards = [
        ('USD Baseline', f"${usd0:,.2f}", 'Opportunity-cost baseline', COLORS['info']),
        ('T-Bill End Value', f"${usd_rf:,.2f}", f"{params['usd_rf_rate_annual']*100:.1f}% for {params['term_days_calendar']}d", COLORS['tbill']),
        ('Break-even Spot', f"{spot_be:.4f}", f"+{be_move:.2f}% from entry", COLORS['breakeven']),
        ('Entry Spread', f"{params['entry_spread']*100:.2f}%", 'From entry vs spot', COLORS['text_secondary']),
        ('USD Baseline', f"${usd0:,.2f}", 'Opportunity-cost baseline', COLORS['info']),
        ('T-Bill End Value', f"${usd_rf:,.2f}", f"{params['usd_rf_rate_annual']*100:.1f}% for {params['term_days_calendar']}d", COLORS['tbill']),
        ('Break-even Spot', f"{spot_be:.4f}", f"+{be_move:.2f}% from entry", COLORS['breakeven']),
        ('Entry Spread', f"{params['entry_spread']*100:.2f}%", 'From entry vs spot', COLORS['text_secondary']),
    ]

    if mc_results:
        # Determine colors based on values
        excess_color = COLORS['profit'] if mc_results['mean_excess'] >= 0 else COLORS['loss']
        sharpe_color = COLORS['profit'] if mc_results['sharpe'] >= 0.3 else (
            COLORS['warning'] if mc_results['sharpe'] >= 0 else COLORS['loss']
        )
        prob_color = COLORS['profit'] if mc_results['prob_underperform_tbill'] < 40 else (
            COLORS['warning'] if mc_results['prob_underperform_tbill'] < 60 else COLORS['loss']
        )

        cards.extend([
            ('Expected Excess', f"{mc_results['mean_excess']:+.2f}%", 'vs T-Bill', excess_color),
            ('Sharpe Ratio', f"{mc_results['sharpe']:.3f}", 'Period excess / std', sharpe_color),
            ('P(Underperform)', f"{mc_results['prob_underperform_tbill']:.1f}%", 'vs T-Bill', prob_color),
            ('VaR 95%', f"{mc_results['var_95_excess']:.2f}%", 'Excess return', COLORS['warning']),
        ])

    html = '<div style="display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0;">'

    for title, value, subtitle, color in cards:
        html += f'''
        <div style="
            background: {COLORS['bg_card']};
            border-left: 4px solid {color};
            padding: 15px 20px;
            border-radius: 8px;
            min-width: 180px;
            flex: 1;
        ">
            <div style="color: {COLORS['text_secondary']}; font-size: 12px; margin-bottom: 5px;">{title}</div>
            <div style="color: {color}; font-size: 24px; font-weight: bold;">{value}</div>
            <div style="color: {COLORS['text_muted']}; font-size: 11px; margin-top: 5px;">{subtitle}</div>
        </div>
        '''

    html += '</div>'
    return html


def create_verdict_html(mc_results: dict, params: dict) -> str:
    """
    Generate verdict HTML box.
    """
    sharpe = mc_results['sharpe']
    prob = mc_results['prob_underperform_tbill']
    mean_excess = mc_results['mean_excess']

    if sharpe >= 0.3 and prob < 40:
        verdict = "FAVORABLE"
        verdict_color = COLORS['profit']
        explanation = "Positive risk-adjusted return with acceptable probability of underperforming T-Bill."
    elif sharpe >= 0 and prob < 55:
        verdict = "MARGINAL"
        verdict_color = COLORS['warning']
        explanation = "Modest risk-adjusted return but elevated probability of underperforming T-Bill."
    else:
        verdict = "UNFAVORABLE"
        verdict_color = COLORS['loss']
        explanation = "Negative risk-adjusted return or high probability of underperforming T-Bill."

    html = f'''
    <div style="
        background: {COLORS['bg_card']};
        border: 2px solid {verdict_color};
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
    ">
        <div style="color: {COLORS['text_secondary']}; font-size: 14px; margin-bottom: 10px;">INVESTMENT VERDICT</div>
        <div style="color: {verdict_color}; font-size: 36px; font-weight: bold; margin-bottom: 15px;">{verdict}</div>
        <div style="color: {COLORS['text_primary']}; font-size: 14px; margin-bottom: 20px;">{explanation}</div>
        <div style="display: flex; justify-content: center; gap: 40px; color: {COLORS['text_secondary']}; font-size: 13px;">
            <div>Sharpe: <span style="color: {verdict_color}; font-weight: bold;">{sharpe:.3f}</span></div>
            <div>P(Underperform): <span style="color: {verdict_color}; font-weight: bold;">{prob:.1f}%</span></div>
            <div>Expected Excess: <span style="color: {verdict_color}; font-weight: bold;">{mean_excess:+.2f}%</span></div>
        </div>
    </div>
    '''
    return html


def export_charts_to_html(
    figs: Dict[str, go.Figure],
    output_path: str,
    title: str = "TRY Carry Trade Analysis"
) -> str:
    """
    Export multiple Plotly figures to a single HTML file.
    """
    html_parts = [f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
                font-family: 'Inter', Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            .chart-container {{
                margin-bottom: 30px;
                background: {COLORS['bg_card']};
                border-radius: 12px;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: {COLORS['text_primary']};
                margin-bottom: 30px;
            }}
            .subtitle {{
                text-align: center;
                color: {COLORS['text_secondary']};
                margin-bottom: 40px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p class="subtitle">Baseline = USD risk-free return over the same horizon</p>
        <p class="subtitle">Baseline = USD risk-free return over the same horizon</p>
    ''']

    for name, fig in figs.items():
        div_id = name.replace(' ', '_').lower()
        html_parts.append(f'''
        <div class="chart-container">
            <div id="{div_id}"></div>
        </div>
        <script>
            Plotly.newPlot("{div_id}", {fig.to_json()});
        </script>
        ''')

    html_parts.append('</body></html>')

    html_content = '\n'.join(html_parts)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def create_realtime_pnl_chart(pnl_data: dict, spot_series: pd.Series) -> go.Figure:
    """
    Create real-time P&L tracking chart.

    Shows:
    - USD/TRY price from entry to now
    - Entry level and break-even level
    - Current P&L annotation
    """
    # Get data from entry to now
    entry_date = pnl_data['entry_date']
    current_date = pnl_data['current_date']

    # Handle timezone - make comparison compatible
    spot_copy = spot_series.copy()
    if spot_copy.index.tz is not None:
        spot_copy.index = spot_copy.index.tz_localize(None)

    # Convert dates to pandas Timestamps (tz-naive) for safe comparison
    entry_ts = pd.Timestamp(entry_date)
    if entry_ts.tz is not None:
        entry_ts = entry_ts.tz_localize(None)
    current_ts = pd.Timestamp(current_date)
    if current_ts.tz is not None:
        current_ts = current_ts.tz_localize(None)

    # Filter spot series
    mask = (spot_copy.index >= entry_ts) & (spot_copy.index <= current_ts)
    trade_period = spot_copy[mask]

    # Fallback if no data in range
    if len(trade_period) == 0:
        trade_period = spot_copy.tail(30)

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=trade_period.index,
        y=trade_period.values,
        mode='lines',
        name='USD/TRY',
        line=dict(color=COLORS['neutral'], width=2),
        fill='tozeroy',
        fillcolor='rgba(116, 185, 255, 0.1)',
    ))

    # Entry spot line
    fig.add_hline(y=pnl_data['entry_spot'], line_dash="dash", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=pnl_data['entry_spot'],
        text=f"Entry: {pnl_data['entry_spot']:.4f}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )

    # Break-even line
    fig.add_hline(y=pnl_data['spot_be'], line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=pnl_data['spot_be'],
        text=f"Break-even: {pnl_data['spot_be']:.4f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )
    fig.add_hline(y=pnl_data['entry_spot'], line_dash="dash", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=pnl_data['entry_spot'],
        text=f"Entry: {pnl_data['entry_spot']:.4f}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )

    # Break-even line
    fig.add_hline(y=pnl_data['spot_be'], line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=pnl_data['spot_be'],
        text=f"Break-even: {pnl_data['spot_be']:.4f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )

    # Current spot marker
    fig.add_trace(go.Scatter(
        x=[current_date],
        y=[pnl_data['current_spot']],
        mode='markers+text',
        marker=dict(size=12, color=COLORS['profit'] if pnl_data['unrealized_pnl_usd'] >= 0 else COLORS['loss']),
        text=[f"Now: {pnl_data['current_spot']:.4f}"],
        textposition='top center',
        name='Current',
    ))

    # P&L color
    pnl_color = COLORS['profit'] if pnl_data['unrealized_pnl_usd'] >= 0 else COLORS['loss']

    # Calculate Y-axis range to include entry, break-even, and current data
    y_min = min(trade_period.min(), pnl_data['entry_spot'], pnl_data['spot_be'], pnl_data['current_spot']) * 0.995
    y_max = max(trade_period.max(), pnl_data['entry_spot'], pnl_data['spot_be'], pnl_data['current_spot']) * 1.005

    fig.update_layout(
        title={
            'text': f"Trade Progress: Entry {entry_date.strftime('%m/%d')} to Now<br>"
                    f"<sub>Unrealized P&L: <span style='color:{pnl_color}'>${pnl_data['unrealized_pnl_usd']:+,.2f} ({pnl_data['unrealized_return_pct']:+.2f}%)</span></sub>",
            'font': {'size': 14},
        },
        xaxis_title='Date',
        yaxis_title='USD/TRY',
        yaxis_range=[y_min, y_max],
        height=400,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def create_trend_chart(spot_series: pd.Series, trend_data: dict, days_remaining: int) -> go.Figure:
    """
    Create trend analysis chart with slope visualization and projection.
    """
    # Recent data (60 days)
    recent = spot_series.tail(60)

    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent.values,
        mode='lines',
        name='USD/TRY',
        line=dict(color=COLORS['neutral'], width=2),
    ))

    # Trend line (linear fit)
    x_numeric = np.arange(len(recent))
    slope = trend_data['slope_daily']
    intercept = recent.iloc[0] - slope * 0  # Start from first point

    # Calculate trend line values
    trend_y = recent.iloc[0] + slope * x_numeric
    fig.add_trace(go.Scatter(
        x=recent.index,
        y=trend_y,
        mode='lines',
        name=f'Trend ({slope:.4f}/day)',
        line=dict(color=COLORS['warning'], width=2, dash='dash'),
    ))

    # Projection to maturity - with interactive daily points
    current_spot = recent.iloc[-1]
    current_date = recent.index[-1]
    projected_spot = trend_data['projected_spot_maturity']
    
    # Generate daily projection points
    projection_days = list(range(0, days_remaining + 1))
    projection_dates = [current_date + pd.Timedelta(days=d) for d in projection_days]
    projection_spots = [current_spot + slope * d for d in projection_days]
    
    fig.add_trace(go.Scatter(
        x=projection_dates,
        y=projection_spots,
        mode='lines+markers',
        name=f'Projection: {projected_spot:.4f}',
        line=dict(color=COLORS['loss'], width=2, dash='dot'),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='Day %{customdata}<br>Date: %{x|%m/%d}<br>Projected: %{y:.4f}<extra></extra>',
        customdata=projection_days,
    ))

    # Acceleration annotation
    accel_color = COLORS['loss'] if trend_data['accel_regime'] == 'ACCELERATING' else (
        COLORS['profit'] if trend_data['accel_regime'] == 'DECELERATING' else COLORS['text_secondary']
    )

    y_min = min(recent.min(), min(projection_spots))
    y_max = max(recent.max(), max(projection_spots))
    padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.1

    fig.update_layout(
        title={
            'text': f"Trend Analysis (R²={trend_data['r_squared']:.3f})<br>"
                    f"<sub>Slope: {trend_data['slope_pct_daily']:.3f}%/day | "
                    f"Regime: <span style='color:{accel_color}'>{trend_data['accel_regime']}</span></sub>",
            'font': {'size': 14},
        },
        xaxis_title='Date',
        yaxis_title='USD/TRY',
        yaxis_range=[y_min - padding, y_max + padding],
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    y_min = min(recent.min(), min(projection_spots))
    y_max = max(recent.max(), max(projection_spots))
    padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.1

    fig.update_layout(
        title={
            'text': f"Trend Analysis (R²={trend_data['r_squared']:.3f})<br>"
                    f"<sub>Slope: {trend_data['slope_pct_daily']:.3f}%/day | "
                    f"Regime: <span style='color:{accel_color}'>{trend_data['accel_regime']}</span></sub>",
            'font': {'size': 14},
        },
        xaxis_title='Date',
        yaxis_title='USD/TRY',
        yaxis_range=[y_min - padding, y_max + padding],
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    # Calculate Y-axis range to include all points (historical, trend, projection)
    all_y_values = list(recent.values) + list(trend_y) + projection_spots
    y_min = min(all_y_values) * 0.995
    y_max = max(all_y_values) * 1.005

    fig.update_layout(
        title={
            'text': f"Trend Analysis (R²={trend_data['r_squared']:.3f})<br>"
                    f"<sub>Slope: {trend_data['slope_pct_daily']:.3f}%/day | "
                    f"Regime: <span style='color:{accel_color}'>{trend_data['accel_regime']}</span></sub>",
            'font': {'size': 14},
        },
        xaxis_title='Date',
        yaxis_title='USD/TRY',
        yaxis_range=[y_min, y_max],
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    return apply_dark_theme(fig)


def create_cushion_gauge(cushion_data: dict) -> go.Figure:
    """
    Create gauge chart for daily cushion vs trend.
    """
    ratio = min(cushion_data['cushion_vs_trend_ratio'], 5)  # Cap at 5x for display

    # Determine color based on status
    if 'DANGER' in cushion_data['status']:
        color = COLORS['loss']
    elif 'WARNING' in cushion_data['status'] or 'CAUTION' in cushion_data['status']:
        color = COLORS['warning']
    else:
        color = COLORS['profit']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        number={'suffix': 'x', 'font': {'size': 40}},
        delta={'reference': 1, 'increasing': {'color': COLORS['profit']}, 'decreasing': {'color': COLORS['loss']}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': COLORS['text_secondary']},
            'bar': {'color': color},
            'bgcolor': COLORS['bg_card'],
            'borderwidth': 2,
            'bordercolor': COLORS['grid'],
            'steps': [
                {'range': [0, 1], 'color': 'rgba(255, 71, 87, 0.25)'},
                {'range': [1, 2], 'color': 'rgba(255, 165, 2, 0.25)'},
                {'range': [2, 5], 'color': 'rgba(0, 210, 106, 0.25)'},
            ],
            'threshold': {
                'line': {'color': COLORS['text_primary'], 'width': 4},
                'thickness': 0.75,
                'value': 1
            }
        },
        title={'text': f"Cushion/Trend Ratio<br><sub>{cushion_data['status']}</sub>",
               'font': {'size': 14}},
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=80, b=30),
    )

    return apply_dark_theme(fig)


def create_cushion_timeline(cushion_data: dict, trend_data: dict) -> go.Figure:
    """
    Create timeline showing cushion erosion at current trend.

    Handles cases where:
    - days_to_be > days_remaining (BE beyond maturity - extend x-axis)
    - days_to_be < days_remaining (BE before maturity - show clearly)
    """
    days_remaining = cushion_data['days_remaining']
    current_spot = cushion_data['spot_current']
    spot_be = cushion_data['spot_be']
    daily_move = trend_data['slope_daily']

    # Generate projection
    days_to_be = cushion_data['days_until_breakeven']
    max_day = days_remaining
    if days_to_be < float('inf') and days_to_be > 0:
        max_day = max(max_day, int(np.ceil(days_to_be)))
    max_day += 5
    days = list(range(0, max_day + 1))
    days_to_be = cushion_data['days_until_breakeven']
    max_day = days_remaining
    if days_to_be < float('inf') and days_to_be > 0:
        max_day = max(max_day, int(np.ceil(days_to_be)))
    max_day += 5
    days = list(range(0, max_day + 1))
    days_remaining = cushion_data['days_remaining']
    current_spot = cushion_data['spot_current']
    spot_be = cushion_data['spot_be']
    daily_move = trend_data['slope_daily']
    days_to_be = cushion_data['days_until_breakeven']

    # Determine x-axis range: extend to include BE day if needed
    if days_to_be < float('inf') and days_to_be > 0:
        # If BE is beyond maturity, extend x-axis to show it
        x_max = max(days_remaining + 5, int(days_to_be) + 5)
    else:
        x_max = days_remaining + 5

    # Generate projection
    days = list(range(0, x_max + 1))
    projected_spots = [current_spot + daily_move * d for d in days]

    fig = go.Figure()

    # Projected path - color based on whether above or below BE
    colors = [COLORS['profit'] if s < spot_be else COLORS['loss'] for s in projected_spots]

    fig.add_trace(go.Scatter(
        x=days,
        y=projected_spots,
        mode='lines+markers',
        name='Projected Spot',
        line=dict(color=COLORS['neutral'], width=2),
        marker=dict(color=colors, size=6),
        hovertemplate='Day %{x}<br>Projected: %{y:.4f}<extra></extra>',
    ))

    # Break-even line
    fig.add_hline(y=spot_be, line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=spot_be,
        text=f"Break-even: {spot_be:.4f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )

    # Maturity line
    fig.add_vline(x=days_remaining, line_dash="dot", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=projected_spots[min(days_remaining, len(projected_spots) - 1)],
        text=f"Maturity: Day {days_remaining}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )

    # Days until BE annotation
    if days_to_be < float('inf') and days_to_be > 0:
        fig.add_vline(x=days_to_be, line_dash="dot", line_color=COLORS['loss'])
        fig.add_annotation(
            xref='paper',
            x=0.98,
            align='right',
            y=spot_be,
            text=f"Hits BE: Day {days_to_be:.0f}",
            showarrow=False,
            font=dict(color=COLORS['loss']),
        )

    # Calculate Y-axis range to include break-even and all projected spots
    y_min = min(min(projected_spots), spot_be, current_spot)
    y_max = max(max(projected_spots), spot_be, current_spot)
    padding = (y_max - y_min) * 0.02 if y_max > y_min else 0.05

    fig.update_layout(
        title={
            'text': f"Cushion Erosion at Current Trend<br>"
                    f"<sub>Daily Cushion: {cushion_data['daily_cushion_pct']:.3f}%/day | "
                    f"Daily Trend: {trend_data['slope_pct_daily']:.3f}%/day</sub>",
            'font': {'size': 14},
        },
        xaxis_title='Days from Now',
        yaxis_title='Projected USD/TRY',
        xaxis_range=[0, max_day],
        yaxis_range=[y_min - padding, y_max + padding],
        height=400,
        showlegend=False,
    )
    fig.add_hline(y=spot_be, line_dash="dash", line_color=COLORS['breakeven'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=spot_be,
        text=f"Break-even: {spot_be:.4f}",
        showarrow=False,
        font=dict(color=COLORS['breakeven']),
    )
    # Break-even line (always visible - key requirement)
    fig.add_hline(
        y=spot_be,
        line_dash="dash",
        line_color=COLORS['breakeven'],
        annotation_text=f"Break-even: {spot_be:.4f}",
        annotation_position="right",
    )

    # Maturity line
    fig.add_vline(x=days_remaining, line_dash="dot", line_color=COLORS['info'])
    fig.add_annotation(
        xref='paper',
        x=0.98,
        align='right',
        y=projected_spots[min(days_remaining, len(projected_spots) - 1)],
        text=f"Maturity: Day {days_remaining}",
        showarrow=False,
        font=dict(color=COLORS['info']),
    )

    # Days until BE annotation
    if days_to_be < float('inf') and days_to_be > 0:
        fig.add_vline(x=days_to_be, line_dash="dot", line_color=COLORS['loss'])
        fig.add_annotation(
            xref='paper',
            x=0.98,
            align='right',
            y=spot_be,
            text=f"Hits BE: Day {days_to_be:.0f}",
            showarrow=False,
            font=dict(color=COLORS['loss']),
        )

    # Calculate Y-axis range to include break-even and all projected spots
    y_min = min(min(projected_spots), spot_be, current_spot)
    y_max = max(max(projected_spots), spot_be, current_spot)
    padding = (y_max - y_min) * 0.02 if y_max > y_min else 0.05

    fig.update_layout(
        title={
            'text': f"Cushion Erosion at Current Trend<br>"
                    f"<sub>Daily Cushion: {cushion_data['daily_cushion_pct']:.3f}%/day | "
                    f"Daily Trend: {trend_data['slope_pct_daily']:.3f}%/day</sub>",
            'font': {'size': 14},
        },
        xaxis_title='Days from Now',
        yaxis_title='Projected USD/TRY',
        xaxis_range=[0, max_day],
        yaxis_range=[y_min - padding, y_max + padding],
        height=400,
        showlegend=False,
    )
    be_annotation = ""
    if days_to_be < float('inf') and days_to_be > 0:
        if days_to_be > days_remaining:
            # BE beyond maturity - still show it
            be_annotation = f" (BE beyond maturity: Day {days_to_be:.0f})"
            fig.add_vline(
                x=days_to_be,
                line_dash="dot",
                line_color=COLORS['loss'],
                annotation_text=f"Hits BE: Day {days_to_be:.0f}",
                annotation_position="top",
            )
        else:
            # BE before maturity - show prominently
            fig.add_vline(
                x=days_to_be,
                line_dash="dot",
                line_color=COLORS['loss'],
                annotation_text=f"DANGER: Hits BE Day {days_to_be:.0f}",
                annotation_position="top",
            )
    elif days_to_be == float('inf') and daily_move <= 0:
        be_annotation = " (Favorable trend - never hits BE)"

    # Calculate Y-axis range to include break-even and all projected spots
    # Ensure adequate padding so lines are never clipped
    all_y_values = projected_spots + [spot_be, current_spot]
    y_min = min(all_y_values) * 0.995
    y_max = max(all_y_values) * 1.005

    fig.update_layout(
        title={
            'text': f"Cushion Erosion at Current Trend<br>"
                    f"<sub>Daily Cushion: {cushion_data['daily_cushion_pct']:.3f}%/day | "
                    f"Daily Trend: {trend_data['slope_pct_daily']:.3f}%/day{be_annotation}</sub>",
            'font': {'size': 14},
        },
        xaxis_title='Days from Now',
        yaxis_title='Projected USD/TRY',
        yaxis_range=[y_min, y_max],
        height=400,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def create_realtime_kpi_html(pnl_data: dict, cushion_data: dict, trend_data: dict) -> str:
    """
    Generate HTML for real-time KPI cards.
    """
    # P&L color
    pnl_color = COLORS['profit'] if pnl_data['unrealized_pnl_usd'] >= 0 else COLORS['loss']
    excess_color = COLORS['profit'] if pnl_data['excess_vs_tbill_now'] >= 0 else COLORS['loss']

    # Cushion color
    if 'DANGER' in cushion_data['status']:
        cushion_color = COLORS['loss']
    elif 'WARNING' in cushion_data['status'] or 'CAUTION' in cushion_data['status']:
        cushion_color = COLORS['warning']
    else:
        cushion_color = COLORS['profit']

    # Trend color
    trend_color = COLORS['loss'] if trend_data['slope_daily'] > 0 else COLORS['profit']

    cards = [
        ('Days Elapsed', f"{pnl_data['days_elapsed']}", f"{pnl_data['days_remaining']} remaining", COLORS['info']),
        ('Unrealized P&L', f"${pnl_data['unrealized_pnl_usd']:+,.2f}", f"{pnl_data['unrealized_return_pct']:+.2f}%", pnl_color),
        ('vs T-Bill', f"${pnl_data['excess_vs_tbill_now']:+,.2f}", 'Excess return', excess_color),
        ('MTM Return', f"{pnl_data['mtm_return_pct']:+.2f}%", 'If spot stays here', pnl_color),
        ('Daily Cushion', f"{cushion_data['daily_cushion_pct']:.3f}%", f"{cushion_data['cushion_vs_trend_ratio']:.1f}x vs trend", cushion_color),
        ('Trend', f"{trend_data['slope_pct_daily']:.3f}%/day", trend_data['accel_regime'], trend_color),
        ('Days to BE', f"{cushion_data['days_until_breakeven']:.1f}", 'At current trend', cushion_color),
        ('Status', cushion_data['status'].split(' - ')[0], cushion_data['status'].split(' - ')[-1] if ' - ' in cushion_data['status'] else '', cushion_color),
    ]

    html = '<div style="display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0;">'

    for title, value, subtitle, color in cards:
        html += f'''
        <div style="
            background: {COLORS['bg_card']};
            border-left: 4px solid {color};
            padding: 12px 16px;
            border-radius: 8px;
            min-width: 150px;
            flex: 1;
        ">
            <div style="color: {COLORS['text_secondary']}; font-size: 11px; margin-bottom: 4px;">{title}</div>
            <div style="color: {color}; font-size: 20px; font-weight: bold;">{value}</div>
            <div style="color: {COLORS['text_muted']}; font-size: 10px; margin-top: 4px;">{subtitle}</div>
        </div>
        '''

    html += '</div>'
    return html


def create_rolling_volatility_chart(vol_df: pd.DataFrame, spot_series: pd.Series) -> go.Figure:
    """
    Create rolling volatility chart with regime classification.

    Args:
        vol_df: DataFrame with 'rolling_vol', 'regime', 'vol_33_threshold', 'vol_67_threshold'
        spot_series: Spot price series for dual-axis

    Returns:
        Plotly figure with rolling vol and regime bands
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Rolling 60-Day Annualized Volatility", "USD/TRY Spot")
    )

    # Get thresholds
    vol_33 = vol_df['vol_33_threshold'].iloc[0] if 'vol_33_threshold' in vol_df.columns else vol_df['rolling_vol'].quantile(0.33)
    vol_67 = vol_df['vol_67_threshold'].iloc[0] if 'vol_67_threshold' in vol_df.columns else vol_df['rolling_vol'].quantile(0.67)

    # Regime colors
    regime_colors = {
        'LOW': COLORS['profit'],
        'MEDIUM': COLORS['warning'],
        'HIGH': COLORS['loss'],
        None: COLORS['neutral'],
    }

    # Rolling volatility line with color by regime
    valid_vol = vol_df['rolling_vol'].dropna()

    for regime in ['LOW', 'MEDIUM', 'HIGH']:
        mask = vol_df['regime'] == regime
        if mask.any():
            regime_data = vol_df[mask]
            fig.add_trace(
                go.Scatter(
                    x=regime_data.index,
                    y=regime_data['rolling_vol'],
                    mode='markers',
                    name=f'{regime} Vol',
                    marker=dict(color=regime_colors[regime], size=4),
                    hovertemplate='%{x}<br>Vol: %{y:.1f}%<br>Regime: ' + regime + '<extra></extra>',
                    showlegend=True,
                ),
                row=1, col=1
            )

    # Add threshold lines
    fig.add_hline(
        y=vol_33,
        line_dash="dash",
        line_color=COLORS['profit'],
        annotation_text=f"LOW/MED: {vol_33:.1f}%",
        annotation_position="right",
        row=1, col=1
    )

    fig.add_hline(
        y=vol_67,
        line_dash="dash",
        line_color=COLORS['loss'],
        annotation_text=f"MED/HIGH: {vol_67:.1f}%",
        annotation_position="right",
        row=1, col=1
    )

    # Current regime annotation
    if len(valid_vol) > 0:
        current_vol = valid_vol.iloc[-1]
        current_regime = vol_df['regime'].iloc[-1]
        fig.add_annotation(
            x=valid_vol.index[-1],
            y=current_vol,
            text=f"Current: {current_vol:.1f}% ({current_regime})",
            showarrow=True,
            arrowhead=2,
            arrowcolor=regime_colors.get(current_regime, COLORS['neutral']),
            font=dict(color=regime_colors.get(current_regime, COLORS['neutral'])),
            row=1, col=1
        )

    # Spot series (bottom panel)
    spot_aligned = spot_series[spot_series.index.isin(vol_df.index)]
    fig.add_trace(
        go.Scatter(
            x=spot_aligned.index,
            y=spot_aligned.values,
            mode='lines',
            name='USD/TRY',
            line=dict(color=COLORS['neutral'], width=1),
            hovertemplate='%{x}<br>Spot: %{y:.4f}<extra></extra>',
            showlegend=False,
        ),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        height=600,
        title={
            'text': "Volatility Regime Analysis<br>"
                    f"<sub>33rd percentile: {vol_33:.1f}% | 67th percentile: {vol_67:.1f}%</sub>",
            'font': {'size': 14},
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )

    fig.update_yaxes(title_text="Annualized Vol (%)", row=1, col=1)
    fig.update_yaxes(title_text="USD/TRY", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return apply_dark_theme(fig)


if __name__ == "__main__":
    print("Plots module loaded successfully.")
    print(f"Available colors: {list(COLORS.keys())}")
