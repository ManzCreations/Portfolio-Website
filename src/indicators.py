"""
Indicator calculations for SYNAPSE web app.
"""

import numpy as np
import pandas as pd
import talib

from .logger import get_logger

logger = get_logger()


class IndicatorCalculator:

    def __init__(self, config):
        self.config = config

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for the decision tree."""
        df = df.copy()
        c = self.config

        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        cl = df['close'].values
        v = df['volume'].values

        # EMAs
        df['ema_9']  = talib.EMA(cl, timeperiod=c.ema_fast_period)
        df['ema_21'] = talib.EMA(cl, timeperiod=c.ema_slow_period)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            cl,
            fastperiod=c.macd_fast,
            slowperiod=c.macd_slow,
            signalperiod=c.macd_signal
        )

        # RSX (TA-Lib doesn't have RSX so we use RSI as proxy)
        df['rsx'] = talib.RSI(cl, timeperiod=c.rsi_period)

        # ROC
        df['roc'] = talib.ROC(cl, timeperiod=c.roc_period)

        # CCI
        df['cci'] = talib.CCI(h, l, cl, timeperiod=c.cci_period)

        # ADX
        df['adx'] = talib.ADX(h, l, cl, timeperiod=c.adx_period)

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            cl,
            timeperiod=c.bb_period,
            nbdevup=c.bb_std,
            nbdevdn=c.bb_std
        )
        df['bb_width']     = df['bb_upper'] - df['bb_lower']
        df['bb_width_sma'] = talib.SMA(
            df['bb_width'].values, timeperiod=c.bb_period
        )

        # ATR
        df['atr'] = talib.ATR(h, l, cl, timeperiod=c.atr_period)

        # Volume SMA
        df['volume_sma'] = talib.SMA(v, timeperiod=c.volume_sma_period)

        # OBV
        df['obv'] = talib.OBV(cl, v)

        # Z-Score
        sma = talib.SMA(cl, timeperiod=c.z_score_period)
        std = talib.STDDEV(cl, timeperiod=c.z_score_period)
        df['z_score'] = np.where(std != 0, (cl - sma) / std, 0)

        logger.info(f"Indicators calculated — {len(df)} candles")
        return df