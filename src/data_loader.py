"""
Data loader for SYNAPSE web app.
Fetch behavior is identical to the original code — only the inputs differ.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .logger import get_logger

logger = get_logger()

MIN_WARMUP_CANDLES = 100


class DataLoader:

    def __init__(self, api_key: str, secret_key: str):
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def fetch(self, config) -> pd.DataFrame:
        if config.range_mode == 'lookback':
            return self._fetch_by_lookback(config)
        else:
            return self._fetch_by_daterange(config)

    def _fetch_by_lookback(self, config) -> pd.DataFrame:
        """
        Exact replica of the original fetch_data method.
        candles_back is used as minutes to go back, identical to original.
        """
        end_date   = datetime.now()
        start_date = end_date - timedelta(minutes=config.lookback)

        logger.info(f"Fetching {config.symbol} [{config.timeframe_str}] "
                    f"from {start_date} to {end_date}")

        return self._request(config.symbol, config.timeframe, start_date, end_date)

    def _fetch_by_daterange(self, config) -> pd.DataFrame:
        """
        Fetch between two explicit datetimes provided by the user.
        Strips timezone to keep naive datetimes consistent with original.
        """
        start_date = datetime.fromisoformat(config.start_datetime).replace(tzinfo=None)
        end_date   = datetime.fromisoformat(config.end_datetime).replace(tzinfo=None)

        logger.info(f"Fetching {config.symbol} [{config.timeframe_str}] "
                    f"from {start_date} to {end_date}")

        return self._request(config.symbol, config.timeframe, start_date, end_date)

    def _request(self, symbol: str, timeframe: TimeFrame,
                 start: datetime, end: datetime) -> pd.DataFrame:
        """
        Identical request logic to the original — no feed override,
        no timezone manipulation.
        """
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end
        )

        bars = self.client.get_stock_bars(request_params)
        df = bars.df

        if df.empty:
            raise ValueError(
                f'No data returned for {symbol}. '
                f'Check that the market was open during the requested period.'
            )

        # Handle multi-index exactly as original
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df = df[df['symbol'] == symbol].copy()
            df.set_index('timestamp', inplace=True)
            df = df.drop('symbol', axis=1)

        # Ensure float64 for TA-Lib, exactly as original
        df = df.astype({
            'open':   np.float64,
            'high':   np.float64,
            'low':    np.float64,
            'close':  np.float64,
            'volume': np.float64
        })

        logger.info(f"Successfully fetched {len(df)} candles")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

        df.sort_index(inplace=True)
        return df

    def validate(self, df: pd.DataFrame, config) -> tuple[bool, str]:
        if df is None or df.empty:
            return False, 'No data returned. Check your symbol and date range.'

        if len(df) < MIN_WARMUP_CANDLES:
            return False, (
                f'Only {len(df)} candles returned — at least {MIN_WARMUP_CANDLES} '
                f'are needed for indicator calculation. '
                f'Try a wider date range or increase your lookback value.'
            )

        if config.timestamp_mode == 'manual' and config.decision_timestamp:
            ts = pd.Timestamp(config.decision_timestamp)
            candles_before = (df.index < ts).sum()
            if candles_before < MIN_WARMUP_CANDLES:
                return False, (
                    f'Only {candles_before} candles exist before the selected '
                    f'timestamp. At least {MIN_WARMUP_CANDLES} are needed. '
                    f'Choose an earlier start date or a later decision timestamp.'
                )

        return True, ''

    def get_decision_index(self, df: pd.DataFrame, config) -> int:
        if config.timestamp_mode == 'latest':
            return len(df) - 1

        ts = pd.Timestamp(config.decision_timestamp)
        mask = df.index <= ts

        if not mask.any():
            raise ValueError(f'No candles found at or before {ts}.')

        # Return the last True position
        return int(np.where(mask)[0][-1])