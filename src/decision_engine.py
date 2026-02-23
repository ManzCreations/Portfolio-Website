"""
Decision engine for SYNAPSE web app.
Each test returns full detail for frontend display.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .logger import get_logger
from .config import Config

logger = get_logger()


class DecisionEngine:

    def __init__(self, config: Config):
        self.config = config

    def make_decision(self, candle: pd.Series, prev_obv: Optional[float] = None) -> Dict:
        """
        Run all 6 layers and return a full result dict including
        per-layer detail for the frontend cards.
        """
        indicators = self._extract(candle)

        if not self._validate(indicators):
            return self._build_result([], 'NO TRADE', 'NONE', 'Insufficient indicator data')

        ema_bullish = indicators['ema_9'] > indicators['ema_21']
        ema_bearish = indicators['ema_9'] < indicators['ema_21']

        layers = [
            self._layer_trend_alignment(indicators, ema_bullish, ema_bearish),
            self._layer_momentum(indicators),
            self._layer_trend_strength(indicators, ema_bullish, ema_bearish),
            self._layer_volatility(indicators),
            self._layer_volume(indicators, prev_obv, ema_bullish, ema_bearish),
            self._layer_statistical(indicators),
        ]

        # Only layers that fired a trade signal
        fired = [l for l in layers if l['result'] == 'TRADE']

        if not fired:
            return self._build_result(layers, 'NO TRADE', 'NONE', 'No conditions met')

        directions = {l['direction'] for l in fired}
        if len(directions) > 1:
            return self._build_result(layers, 'NO TRADE', 'NONE', 'Conflicting signals')

        direction = directions.pop()
        reason    = ', '.join(l['reason'] for l in fired)
        return self._build_result(layers, 'TRADE', direction, reason, indicators)

    # ------------------------------------------------------------------ #
    # LAYER BUILDERS
    # ------------------------------------------------------------------ #

    def _layer_trend_alignment(self, ind, ema_bull, ema_bear):
        macd_bull  = ind['macd'] > ind['macd_signal'] and ind['macd'] > 0
        macd_bear  = ind['macd'] < ind['macd_signal'] and ind['macd'] < 0
        vwap_bull  = ind['close'] > ind['vwap']
        vwap_bear  = ind['close'] < ind['vwap']

        if ema_bull and macd_bull and vwap_bull:
            result, direction, reason = 'TRADE', 'LONG', 'Trend Direction Aligned (Bullish)'
        elif ema_bear and macd_bear and vwap_bear:
            result, direction, reason = 'TRADE', 'SHORT', 'Trend Direction Aligned (Bearish)'
        else:
            result, direction, reason = 'NO TRADE', 'NONE', 'Trend not aligned'

        return {
            'layer':     1,
            'name':      'Trend Alignment',
            'result':    result,
            'direction': direction,
            'reason':    reason,
            'indicators': {
                'EMA 9':        f"{ind['ema_9']:.2f}",
                'EMA 21':       f"{ind['ema_21']:.2f}",
                'MACD':         f"{ind['macd']:.4f}",
                'MACD Signal':  f"{ind['macd_signal']:.4f}",
                'Close':        f"{ind['close']:.2f}",
                'VWAP':         f"{ind['vwap']:.2f}",
            },
            'long_condition':  'EMA9 > EMA21  AND  MACD > Signal AND > 0  AND  Close > VWAP',
            'short_condition': 'EMA9 < EMA21  AND  MACD < Signal AND < 0  AND  Close < VWAP',
        }

    def _layer_momentum(self, ind):
        rsi_os   = ind['rsi'] < self.config.rsi_oversold
        rsi_ob   = ind['rsi'] > self.config.rsi_overbought
        roc_pos  = ind['roc'] > self.config.roc_strong_threshold
        roc_neg  = ind['roc'] < -self.config.roc_strong_threshold
        cci_bull = ind['cci'] > self.config.cci_threshold
        cci_bear = ind['cci'] < -self.config.cci_threshold

        if (rsi_os and roc_pos) or cci_bull:
            result, direction, reason = 'TRADE', 'LONG', 'Momentum Quality Strong (Bullish)'
        elif (rsi_ob and roc_neg) or cci_bear:
            result, direction, reason = 'TRADE', 'SHORT', 'Momentum Quality Strong (Bearish)'
        else:
            result, direction, reason = 'NO TRADE', 'NONE', 'Momentum not strong'

        return {
            'layer':     2,
            'name':      'Momentum Quality',
            'result':    result,
            'direction': direction,
            'reason':    reason,
            'indicators': {
                'RSI':  f"{ind['rsi']:.2f}",
                'ROC':  f"{ind['roc']:.2f}",
                'CCI':  f"{ind['cci']:.2f}",
            },
            'long_condition':  f"(RSI < {self.config.rsi_oversold} AND ROC > {self.config.roc_strong_threshold})  OR  CCI > {self.config.cci_threshold}",
            'short_condition': f"(RSI > {self.config.rsi_overbought} AND ROC < -{self.config.roc_strong_threshold})  OR  CCI < -{self.config.cci_threshold}",
        }

    def _layer_trend_strength(self, ind, ema_bull, ema_bear):
        strong = ind['adx'] >= self.config.adx_threshold

        if strong and ema_bull:
            result, direction, reason = 'TRADE', 'LONG',  'Trend Strength >= 25 (Bullish)'
        elif strong and ema_bear:
            result, direction, reason = 'TRADE', 'SHORT', 'Trend Strength >= 25 (Bearish)'
        else:
            result, direction, reason = 'NO TRADE', 'NONE', 'Weak trend'

        return {
            'layer':     3,
            'name':      'Trend Strength',
            'result':    result,
            'direction': direction,
            'reason':    reason,
            'indicators': {
                'ADX':   f"{ind['adx']:.2f}",
                'EMA 9': f"{ind['ema_9']:.2f}",
                'EMA 21':f"{ind['ema_21']:.2f}",
            },
            'long_condition':  f"ADX >= {self.config.adx_threshold}  AND  EMA9 > EMA21",
            'short_condition': f"ADX >= {self.config.adx_threshold}  AND  EMA9 < EMA21",
        }

    def _layer_volatility(self, ind):
        expanding = ind['bb_width'] > ind['bb_width_sma'] * self.config.bb_width_expansion_factor
        above_mid = ind['close'] > ind['bb_middle'] and ind['close'] < ind['bb_upper']
        below_mid = ind['close'] < ind['bb_middle'] and ind['close'] > ind['bb_lower']

        if expanding and above_mid:
            result, direction, reason = 'TRADE', 'LONG',  'Volatility Expanding (Bullish)'
        elif expanding and below_mid:
            result, direction, reason = 'TRADE', 'SHORT', 'Volatility Expanding (Bearish)'
        else:
            result, direction, reason = 'NO TRADE', 'NONE', 'Volatility not expanding'

        return {
            'layer':     4,
            'name':      'Volatility Expansion',
            'result':    result,
            'direction': direction,
            'reason':    reason,
            'indicators': {
                'BB Width':     f"{ind['bb_width']:.4f}",
                'BB Width SMA': f"{ind['bb_width_sma']:.4f}",
                'BB Upper':     f"{ind['bb_upper']:.2f}",
                'BB Middle':    f"{ind['bb_middle']:.2f}",
                'BB Lower':     f"{ind['bb_lower']:.2f}",
                'Close':        f"{ind['close']:.2f}",
            },
            'long_condition':  f"BB Width > BB Width SMA × {self.config.bb_width_expansion_factor}  AND  Middle < Close < Upper",
            'short_condition': f"BB Width > BB Width SMA × {self.config.bb_width_expansion_factor}  AND  Lower < Close < Middle",
        }

    def _layer_volume(self, ind, prev_obv, ema_bull, ema_bear):
        vol_good = ind['volume'] > ind['volume_sma'] * self.config.volume_participation_factor
        obv_inc  = prev_obv is not None and ind['obv'] > prev_obv

        if vol_good and prev_obv is not None:
            if obv_inc and ema_bull:
                result, direction, reason = 'TRADE', 'LONG',  'Volume Participation Good (Bullish)'
            elif not obv_inc and ema_bear:
                result, direction, reason = 'TRADE', 'SHORT', 'Volume Participation Good (Bearish)'
            else:
                result, direction, reason = 'NO TRADE', 'NONE', 'Volume direction conflicts with trend'
        else:
            result, direction, reason = 'NO TRADE', 'NONE', 'Volume not sufficient'

        obv_prev_str = f"{prev_obv:,.0f}" if prev_obv is not None else 'N/A'
        return {
            'layer':     5,
            'name':      'Volume Participation',
            'result':    result,
            'direction': direction,
            'reason':    reason,
            'indicators': {
                'Volume':     f"{ind['volume']:,.0f}",
                'Volume SMA': f"{ind['volume_sma']:,.0f}",
                'OBV':        f"{ind['obv']:,.0f}",
                'Prev OBV':   obv_prev_str,
            },
            'long_condition':  f"Volume > Volume SMA × {self.config.volume_participation_factor}  AND  OBV rising  AND  EMA9 > EMA21",
            'short_condition': f"Volume > Volume SMA × {self.config.volume_participation_factor}  AND  OBV falling  AND  EMA9 < EMA21",
        }

    def _layer_statistical(self, ind):
        z = ind['z_score']
        if z > self.config.z_score_extreme_threshold:
            result, direction, reason = 'TRADE', 'LONG',  'Statistical Edge Extreme (Bullish)'
        elif z < -self.config.z_score_extreme_threshold:
            result, direction, reason = 'TRADE', 'SHORT', 'Statistical Edge Extreme (Bearish)'
        else:
            result, direction, reason = 'NO TRADE', 'NONE', 'No statistical extreme'

        return {
            'layer':     6,
            'name':      'Statistical Edge',
            'result':    result,
            'direction': direction,
            'reason':    reason,
            'indicators': {
                'Z-Score':   f"{z:.4f}",
                'ATR':       f"{ind['atr']:.4f}",
            },
            'long_condition':  f"Z-Score > {self.config.z_score_extreme_threshold}",
            'short_condition': f"Z-Score < -{self.config.z_score_extreme_threshold}",
        }

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    def _extract(self, candle: pd.Series) -> Dict:
        return {
            'ema_9':        candle['ema_9'],
            'ema_21':       candle['ema_21'],
            'macd':         candle['macd'],
            'macd_signal':  candle['macd_signal'],
            'close':        candle['close'],
            'vwap':         candle['vwap'],
            'rsi':          candle['rsx'],
            'roc':          candle['roc'],
            'cci':          candle['cci'],
            'adx':          candle['adx'],
            'bb_width':     candle['bb_width'],
            'bb_width_sma': candle['bb_width_sma'],
            'bb_upper':     candle['bb_upper'],
            'bb_middle':    candle['bb_middle'],
            'bb_lower':     candle['bb_lower'],
            'volume':       candle['volume'],
            'volume_sma':   candle['volume_sma'],
            'z_score':      candle['z_score'],
            'atr':          candle['atr'],
            'obv':          candle['obv'],
        }

    def _validate(self, ind: Dict) -> bool:
        for key in ['ema_9', 'ema_21', 'adx', 'atr']:
            if pd.isna(ind[key]):
                return False
        return True

    def _build_result(self, layers, decision, direction, reason, indicators=None) -> Dict:
        result = {
            'decision':  decision,
            'direction': direction,
            'reason':    reason,
            'signal':    1 if direction == 'LONG' else (-1 if direction == 'SHORT' else 0),
            'layers':    layers,
        }
        if indicators:
            result['close'] = indicators['close']
            result['atr']   = indicators['atr']
            result['z_score'] = indicators['z_score']
        return result