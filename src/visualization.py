"""
Interactive TradingView/Alpaca-style chart builder using Plotly.
- Candlesticks + VWAP on main panel
- Volume as sub-axis below candles (same panel)
- Greyed background for non-market hours
- Current price ticker with change indicator
- Buy / Sell signal markers at every signaled candle
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .logger import get_logger

logger = get_logger()

MARKET_OPEN  = 9 * 60 + 30   # 9:30 AM in minutes since midnight
MARKET_CLOSE = 16 * 60        # 4:00 PM in minutes since midnight


# ------------------------------------------------------------------ #
# SIGNAL COMPUTATION — called once in app.py after indicators are ready
# ------------------------------------------------------------------ #

def compute_all_signals(df: pd.DataFrame, decision_engine, risk_manager) -> pd.DataFrame:
    """
    Run the decision engine on every candle in df and return a signals DataFrame.

    Columns returned:
        signal_type  : 'LONG', 'SHORT', or None
        low          : candle low  (used to place BUY marker below wick)
        high         : candle high (used to place SELL marker above wick)
        close        : close price at that candle
        stop_loss    : risk param (NaN if no trade)
        take_profit  : risk param (NaN if no trade)

    Usage in app.py:
        signals = compute_all_signals(df, engine, risk_mgr)
        fig_dict = build_chart(df, symbol, decision_idx, signals=signals)
    """
    records  = []
    prev_obv = None

    for i in range(len(df)):
        candle   = df.iloc[i]
        obv_val  = candle.get('obv', np.nan)
        decision = decision_engine.make_decision(candle, prev_obv)
        prev_obv = float(obv_val) if not pd.isna(obv_val) else prev_obv

        if decision['decision'] == 'TRADE':
            decision = risk_manager.calculate_risk_parameters(decision)
            records.append({
                'signal_type':  decision['direction'],
                'low':          float(candle['low']),
                'high':         float(candle['high']),
                'close':        decision.get('close', np.nan),
                'stop_loss':    decision.get('stop_loss', np.nan),
                'take_profit':  decision.get('take_profit', np.nan),
                'partial_exit_1': decision.get('partial_exit_1', np.nan),
                'partial_exit_2': decision.get('partial_exit_2', np.nan),
                'risk_reward_ratio': decision.get('risk_reward_ratio', np.nan),
                'reason':       decision.get('reason', ''),
                'layers':       decision.get('layers', []),   # full 6-layer breakdown
            })
        else:
            records.append({
                'signal_type':  None,
                'low':          float(candle['low']),
                'high':         float(candle['high']),
                'close':        np.nan,
                'stop_loss':    np.nan,
                'take_profit':  np.nan,
                'partial_exit_1': np.nan,
                'partial_exit_2': np.nan,
                'risk_reward_ratio': np.nan,
                'reason':       '',
                'layers':       [],
            })

    return pd.DataFrame(records, index=df.index)


# ------------------------------------------------------------------ #
# CHART BUILDER
# ------------------------------------------------------------------ #

def build_chart(df: pd.DataFrame, symbol: str, decision_idx: int = None,
                signals: pd.DataFrame = None) -> dict:
    """
    Build an Alpaca-style interactive chart.

    Parameters:
        df           : OHLCV DataFrame with indicators, indexed by timestamp
        symbol       : Ticker string e.g. 'SPY'
        decision_idx : Integer index of the selected decision candle
        signals      : Optional DataFrame from compute_all_signals(). When provided,
                       green BUY labels are drawn below each LONG candle and red SELL
                       labels are drawn above each SHORT candle.
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
    # 6. BUY / SELL SIGNAL MARKERS
    # ------------------------------------------------------------------ #
    if signals is not None and not signals.empty:
        _add_signal_markers(fig, timestamps, df, signals)

    # ------------------------------------------------------------------ #
    # 7. NON-MARKET HOURS — grey shading
    # ------------------------------------------------------------------ #
    shapes = _build_market_shapes(df)

    # ------------------------------------------------------------------ #
    # 8. DECISION CANDLE vertical line
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
    # 9. CURRENT PRICE LINE
    # ------------------------------------------------------------------ #
    last_close  = float(df['close'].iloc[-1])
    first_close = float(df['close'].iloc[0])
    price_change     = last_close - first_close
    price_change_pct = (price_change / first_close) * 100
    price_color      = '#26a69a' if price_change >= 0 else '#ef5350'

    shapes.append(dict(
        type='line',
        x0=0, x1=len(df) - 1,
        y0=last_close, y1=last_close,
        xref='x', yref='y',
        line=dict(color=price_color, width=1, dash='dot')
    ))

    # ------------------------------------------------------------------ #
    # 10. LAYOUT
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
    # 11. AXIS STYLING
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
        range=[0, max_vol * 5],
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


# ------------------------------------------------------------------ #
# SIGNAL MARKER HELPER
# ------------------------------------------------------------------ #

def _add_signal_markers(fig, timestamps: list, df: pd.DataFrame,
                        signals: pd.DataFrame) -> None:
    """
    Add BUY and SELL scatter traces with triangle markers and colored pill labels.

    BUY  — green triangle pointing up, placed below the candle low wick
    SELL — red triangle pointing down, placed above the candle high wick

    An ATR-based offset keeps labels from overlapping wicks. Falls back to
    0.2% of price if ATR is not in the DataFrame.
    """
    # Offset so markers clear the wicks — half ATR or 0.2% fallback
    if 'atr' in df.columns:
        atr_series = df['atr'].ffill().fillna(df['close'] * 0.002)
        offset     = (atr_series * 0.5).values
    else:
        offset = (df['close'] * 0.002).values

    import json

    def _signal_customdata(row):
        """Serialize full signal detail to a JSON string for the frontend click handler."""
        return json.dumps({
            'signal_type':       row['signal_type'],
            'close':             round(row['close'], 4)             if not pd.isna(row['close'])             else None,
            'stop_loss':         round(row['stop_loss'], 4)         if not pd.isna(row['stop_loss'])         else None,
            'take_profit':       round(row['take_profit'], 4)       if not pd.isna(row['take_profit'])       else None,
            'partial_exit_1':    round(row['partial_exit_1'], 4)    if not pd.isna(row['partial_exit_1'])    else None,
            'partial_exit_2':    round(row['partial_exit_2'], 4)    if not pd.isna(row['partial_exit_2'])    else None,
            'risk_reward_ratio': round(row['risk_reward_ratio'], 2) if not pd.isna(row['risk_reward_ratio']) else None,
            'reason':            row['reason'],
            'layers':            row['layers'],   # full list of layer dicts
        })

    # ---- BUY markers ------------------------------------------------- #
    buy_mask = signals['signal_type'] == 'LONG'
    if buy_mask.any():
        buy_indices  = np.where(buy_mask.values)[0]
        buy_x        = [timestamps[i] for i in buy_indices]
        buy_y        = [float(signals['low'].iloc[i]) - offset[i] for i in buy_indices]
        buy_custom   = [_signal_customdata(signals.iloc[i]) for i in buy_indices]
        buy_hover    = [
            f"<b>BUY</b>  ${signals['close'].iloc[i]:.2f}<br><i>Enable Inspector Mode (ℹ︎ button) then click to view detail</i>"
            for i in buy_indices
        ]

        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode='markers',
                name='BUY',
                marker=dict(
                    symbol='triangle-up',
                    size=13,
                    color='#26a69a',
                    line=dict(color='#1a7a6e', width=1),
                ),
                hovertemplate='%{text}<extra></extra>',
                text=buy_hover,
                customdata=buy_custom,
                hoverlabel=dict(
                    bgcolor='#26a69a',
                    bordercolor='#1a7a6e',
                    font=dict(color='white', size=11),
                ),
            ),
            secondary_y=False
        )

        for x, y in zip(buy_x, buy_y):
            fig.add_annotation(
                x=x, y=y,
                text='<b>BUY</b>',
                showarrow=False,
                yshift=-16,
                font=dict(size=9, color='#080808', family='Poppins, sans-serif'),
                bgcolor='#26a69a',
                bordercolor='#1a7a6e',
                borderwidth=1,
                borderpad=3,
                opacity=0.95,
            )

    # ---- SELL markers ------------------------------------------------ #
    sell_mask = signals['signal_type'] == 'SHORT'
    if sell_mask.any():
        sell_indices = np.where(sell_mask.values)[0]
        sell_x       = [timestamps[i] for i in sell_indices]
        sell_y       = [float(signals['high'].iloc[i]) + offset[i] for i in sell_indices]
        sell_custom  = [_signal_customdata(signals.iloc[i]) for i in sell_indices]
        sell_hover   = [
            f"<b>SELL</b>  ${signals['close'].iloc[i]:.2f}<br><i>Enable Inspector Mode (ℹ︎ button) then click to view detail</i>"
            for i in sell_indices
        ]

        fig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode='markers',
                name='SELL',
                marker=dict(
                    symbol='triangle-down',
                    size=13,
                    color='#ef5350',
                    line=dict(color='#b71c1c', width=1),
                ),
                hovertemplate='%{text}<extra></extra>',
                text=sell_hover,
                customdata=sell_custom,
                hoverlabel=dict(
                    bgcolor='#ef5350',
                    bordercolor='#b71c1c',
                    font=dict(color='white', size=11),
                ),
            ),
            secondary_y=False
        )

        for x, y in zip(sell_x, sell_y):
            fig.add_annotation(
                x=x, y=y,
                text='<b>SELL</b>',
                showarrow=False,
                yshift=16,
                font=dict(size=9, color='white', family='Poppins, sans-serif'),
                bgcolor='#ef5350',
                bordercolor='#b71c1c',
                borderwidth=1,
                borderpad=3,
                opacity=0.95,
            )

    # ---- Exit level lines -------------------------------------------- #
    _add_exit_levels(fig, timestamps, signals)


# ------------------------------------------------------------------ #
# EXIT LEVEL LINES
# ------------------------------------------------------------------ #

def _add_exit_levels(fig, timestamps: list, signals: 'pd.DataFrame') -> None:
    """
    For every BUY or SELL signal draw four horizontal dotted lines extending
    rightward from the entry candle to the next signal (or end of data):

        Stop Loss      -- red dotted,    opacity 0.75
        Partial Exit 1 -- orange dotted, opacity 0.65
        Partial Exit 2 -- orange dotted, opacity 0.65
        Take Profit    -- green dotted,  opacity 0.75

    Lines are shapes (not traces) so they don't appear in the legend.
    A small price label annotation sits at the right end of each line.
    """
    import pandas as pd

    n = len(timestamps)

    # Integer positions of every signal candle
    signal_indices = signals.index[signals['signal_type'].notna()].tolist()
    signal_pos     = [signals.index.get_loc(idx) for idx in signal_indices]

    def end_pos(pos):
        """Rightmost candle index this signal's lines extend to."""
        later = [p for p in signal_pos if p > pos]
        return later[0] if later else n - 1

    level_cfg = [
        ('stop_loss',      '#ef5350', 0.75, 'SL'),
        ('partial_exit_1', '#FF9800', 0.65, 'PE1'),
        ('partial_exit_2', '#FF9800', 0.65, 'PE2'),
        ('take_profit',    '#26a69a', 0.75, 'TP'),
    ]

    for sig_idx in signal_indices:
        row = signals.loc[sig_idx]
        x0  = signals.index.get_loc(sig_idx)
        x1  = end_pos(x0)
        ts_x1 = timestamps[x1]

        for col, color, opacity, label in level_cfg:
            price = row[col]
            if pd.isna(price):
                continue

            # Horizontal dotted line
            fig.add_shape(
                type='line',
                x0=x0, x1=x1,
                y0=price, y1=price,
                xref='x', yref='y',
                line=dict(color=color, width=1, dash='dot'),
                opacity=opacity,
            )

            # Price label at right end
            fig.add_annotation(
                x=ts_x1,
                y=price,
                text=f'<b>{label}</b> ${price:.2f}',
                showarrow=False,
                xanchor='left',
                xshift=6,
                font=dict(size=9, color=color, family='Poppins, sans-serif'),
                bgcolor='rgba(15,30,45,0.8)',
                borderpad=2,
                opacity=0.9,
            )


# ------------------------------------------------------------------ #
# MARKET HOURS SHADING
# ------------------------------------------------------------------ #

def _build_market_shapes(df: pd.DataFrame) -> list:
    """
    Build grey rectangle shapes covering non-market hours.
    Uses integer index positions to work correctly with category x-axis.
    """
    shapes     = []
    timestamps = df.index

    if timestamps.empty:
        return shapes

    for i, ts in enumerate(timestamps):
        minutes_since_midnight = ts.hour * 60 + ts.minute

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


# ------------------------------------------------------------------ #
# DECISION LINE UPDATE (used by /api/decision after chart click)
# ------------------------------------------------------------------ #

def get_decision_line_update(decision_idx: int, n_candles: int) -> dict:
    """
    Returns a Plotly relayout update dict that moves the decision
    vertical line to a new candle index.
    """
    return {
        'decision_idx': decision_idx,
    }