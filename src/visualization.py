"""
Interactive TradingView/Alpaca-style chart builder using Plotly.
- Candlesticks + VWAP on main panel
- Volume as sub-axis below candles (same panel)
- Greyed background for non-market hours
- Current price ticker with change indicator
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .logger import get_logger

logger = get_logger()

MARKET_OPEN  = 9 * 60 + 30   # 9:30 AM in minutes since midnight
MARKET_CLOSE = 16 * 60        # 4:00 PM in minutes since midnight


def build_chart(df: pd.DataFrame, symbol: str, decision_idx: int = None) -> dict:
    """
    Build an Alpaca-style interactive chart with:
      - Candlesticks + VWAP on main panel
      - Volume bars sharing the same x-axis below candles (secondary y-axis)
      - Grey shading for non-market hours
      - Current price bar with change/percent
    """
    logger.info(f"Building chart for {symbol} — {len(df)} candles")

    # ------------------------------------------------------------------ #
    # 1. SETUP — two y-axes on one panel
    # ------------------------------------------------------------------ #
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    timestamps = df.index.astype(str).tolist()

    # ------------------------------------------------------------------ #
    # 2. VOLUME BARS — drawn first so candles render on top
    # ------------------------------------------------------------------ #
    vol_colors = [
        'rgba(38,166,154,0.3)' if c >= o else 'rgba(239,83,80,0.3)'
        for c, o in zip(df['close'], df['open'])
    ]

    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=df['volume'],
            name='Volume',
            marker_color=vol_colors,
            marker_line_width=0,
            hovertemplate='Vol: %{y:,.0f}<extra></extra>',
            showlegend=True
        ),
        secondary_y=True
    )

    # Volume SMA
    if 'volume_sma' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df['volume_sma'],
                name='Vol SMA 20',
                line=dict(color='rgba(255,152,0,0.6)', width=1),
                hovertemplate='Vol SMA: %{y:,.0f}<extra></extra>'
            ),
            secondary_y=True
        )

    # ------------------------------------------------------------------ #
    # 3. CANDLESTICKS — primary y-axis
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing=dict(line=dict(color='#26a69a', width=1), fillcolor='#26a69a'),
            decreasing=dict(line=dict(color='#ef5350', width=1), fillcolor='#ef5350'),
            whiskerwidth=0.3,
            hoverinfo='x+y'
        ),
        secondary_y=False
    )

    # ------------------------------------------------------------------ #
    # 4. VWAP
    # ------------------------------------------------------------------ #
    if 'vwap' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df['vwap'],
                name='VWAP',
                line=dict(color='#FF9800', width=1.5),
                hovertemplate='VWAP: %{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )

    # ------------------------------------------------------------------ #
    # 5. EMA overlays (hidden by default, toggle via legend)
    # ------------------------------------------------------------------ #
    if 'ema_9' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=df['ema_9'],
                name='EMA 9',
                line=dict(color='#00d5ff', width=1),
                hovertemplate='EMA9: %{y:.2f}<extra></extra>',
                visible='legendonly'
            ),
            secondary_y=False
        )

    if 'ema_21' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=df['ema_21'],
                name='EMA 21',
                line=dict(color='#7B61FF', width=1),
                hovertemplate='EMA21: %{y:.2f}<extra></extra>',
                visible='legendonly'
            ),
            secondary_y=False
        )

    # ------------------------------------------------------------------ #
    # 6. NON-MARKET HOURS — grey shading
    # ------------------------------------------------------------------ #
    shapes = _build_market_shapes(df)

    # ------------------------------------------------------------------ #
    # 7. DECISION CANDLE vertical line
    # ------------------------------------------------------------------ #
    if decision_idx is not None and 0 <= decision_idx < len(df):
        shapes.append(dict(
            type='line',
            x0=decision_idx, x1=decision_idx,
            y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash')
        ))
        fig.add_annotation(
            x=decision_idx,
            y=1.01,
            yref='paper',
            text='Decision',
            showarrow=False,
            font=dict(color='white', size=10),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1
        )

    # ------------------------------------------------------------------ #
    # 8. CURRENT PRICE LINE
    # ------------------------------------------------------------------ #
    last_close = float(df['close'].iloc[-1])
    first_close = float(df['close'].iloc[0])
    price_change = last_close - first_close
    price_change_pct = (price_change / first_close) * 100
    price_color = '#26a69a' if price_change >= 0 else '#ef5350'

    shapes.append(dict(
        type='line',
        x0=0, x1=len(df) - 1,
        y0=last_close, y1=last_close,
        xref='x', yref='y',
        line=dict(color=price_color, width=1, dash='dot')
    ))

    # ------------------------------------------------------------------ #
    # 9. LAYOUT
    # ------------------------------------------------------------------ #
    change_arrow = '▲' if price_change >= 0 else '▼'
    change_sign  = '+' if price_change >= 0 else ''

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f1e2d',
        plot_bgcolor='#131722',
        font=dict(family='Poppins, sans-serif', color='rgba(255,255,255,0.7)', size=11),

        title=dict(
            text=(
                f'<b>{symbol}</b>'
                f'<span style="font-size:18px; color:white; margin-left:10px">'
                f'  ${last_close:.2f}</span>'
                f'<span style="font-size:14px; color:{price_color}; margin-left:8px">'
                f'  {change_arrow} {change_sign}{price_change:.2f} '
                f'({change_sign}{price_change_pct:.2f}%)</span>'
            ),
            font=dict(size=14, color='white'),
            x=0.01
        ),

        xaxis=dict(rangeslider=dict(visible=False)),

        legend=dict(
            orientation='h',
            x=0, y=1.06,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        ),

        shapes=shapes,
        margin=dict(l=10, r=80, t=60, b=40),
        height=600,

        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e2d3d',
            bordercolor='rgba(0,213,255,0.3)',
            font=dict(size=11, color='white')
        ),

        dragmode='pan',
        modebar=dict(
            bgcolor='rgba(0,0,0,0)',
            color='rgba(255,255,255,0.4)',
            activecolor='#00d5ff',
        )
    )

    # ------------------------------------------------------------------ #
    # 10. AXIS STYLING
    # ------------------------------------------------------------------ #
    axis_style = dict(
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)',
        tickfont=dict(size=10),
        showspikes=True,
        spikecolor='rgba(255,255,255,0.3)',
        spikethickness=1,
        spikedash='dot'
    )

    # Primary y-axis — price, right side
    fig.update_yaxes(
        axis_style,
        secondary_y=False,
        side='right',
        title_text='Price',
        showgrid=True
    )

    # Secondary y-axis — volume, scaled so bars sit in bottom 20% of chart
    max_vol = df['volume'].max()
    fig.update_yaxes(
        axis_style,
        secondary_y=True,
        side='right',
        showgrid=False,
        showticklabels=False,
        range=[0, max_vol * 5],   # pushes volume bars to bottom 20%
        overlaying='y'
    )

    fig.update_xaxes(
        axis_style,
        type='category', 
        tickangle=-30,
        tickfont=dict(size=9),
        showspikes=True,
        nticks=12 
    )

    return fig.to_dict()


def _build_market_shapes(df: pd.DataFrame) -> list:
    """
    Build grey rectangle shapes covering non-market hours.
    Uses integer index positions to work correctly with category x-axis.
    """
    shapes = []
    timestamps = df.index

    if timestamps.empty:
        return shapes

    for i, ts in enumerate(timestamps):
        minutes_since_midnight = ts.hour * 60 + ts.minute

        # Outside market hours: before 9:30 or at/after 16:00
        if minutes_since_midnight < MARKET_OPEN or minutes_since_midnight >= MARKET_CLOSE:
            shapes.append(dict(
                type='rect',
                x0=i - 0.5, x1=i + 0.5,
                y0=0, y1=1,
                xref='x', yref='paper',
                fillcolor='rgba(0,0,0,0.35)',
                line=dict(width=0),
                layer='below'
            ))

    return shapes


def get_decision_line_update(decision_idx: int, n_candles: int) -> dict:
    """
    Returns a Plotly relayout update dict that moves the decision
    vertical line to a new candle index. Called after click recalculation
    so we don't rebuild the entire figure.
    """
    # Find the shape index for the decision line (last shape added)
    # We return the full shapes list replacement so Plotly can update cleanly
    return {
        'decision_idx': decision_idx,
    }