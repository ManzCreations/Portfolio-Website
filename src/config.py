"""
Configuration for SYNAPSE web app.
Built from the request payload instead of .env file.
"""

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class Config:
    """Holds all strategy parameters for a single analysis request."""

    # Timeframe mapping
    TIMEFRAME_MAP = {
        '1Min':  TimeFrame.Minute,
        '2Min':  TimeFrame(2,  TimeFrameUnit.Minute),
        '3Min':  TimeFrame(3,  TimeFrameUnit.Minute),
        '5Min':  TimeFrame(5,  TimeFrameUnit.Minute),
        '10Min': TimeFrame(10, TimeFrameUnit.Minute),
        '15Min': TimeFrame(15, TimeFrameUnit.Minute),
        '30Min': TimeFrame(30, TimeFrameUnit.Minute),
        '1Hour': TimeFrame.Hour,
        '2Hour': TimeFrame(2,  TimeFrameUnit.Hour),
        '4Hour': TimeFrame(4,  TimeFrameUnit.Hour),
        '1Day':  TimeFrame.Day,
        '1Week': TimeFrame.Week,
    }

    # Timeframe to minutes mapping
    TIMEFRAME_MINUTES = {
        '1Min':  1,
        '2Min':  2,
        '3Min':  3,
        '5Min':  5,
        '10Min': 10,
        '15Min': 15,
        '30Min': 30,
        '1Hour': 60,
        '2Hour': 120,
        '4Hour': 240,
        '1Day':  390,   # ~6.5 hours of trading per day
        '1Week': 1950,  # 5 trading days
    }

    def __init__(self, payload: dict):
        """
        Initialize from the JSON payload sent by the frontend.

        Expected payload keys:
            api_key, secret_key, symbol, timeframe,
            range_mode ('lookback' or 'daterange'),
            lookback (int, if range_mode == 'lookback'),
            start_datetime (str, if range_mode == 'daterange'),
            end_datetime   (str, if range_mode == 'daterange'),
            timestamp_mode ('latest' or 'manual'),
            decision_timestamp (str, if timestamp_mode == 'manual')
        """

        # Credentials
        self.api_key    = payload.get('api_key', '')
        self.secret_key = payload.get('secret_key', '')

        # Trading parameters
        self.symbol        = payload.get('symbol', 'SPY').upper()
        self.timeframe_str = payload.get('timeframe', '1Min')

        # Map timeframe string to Alpaca TimeFrame object
        self.timeframe = self.TIMEFRAME_MAP.get(self.timeframe_str, TimeFrame.Minute)

        # Data range
        self.range_mode = payload.get('range_mode', 'lookback')
        self.candles    = int(payload.get('lookback', 500))

        # Convert candles → minutes for the fetch call
        mins_per_candle     = self.TIMEFRAME_MINUTES.get(self.timeframe_str, 1)
        # Add 40% buffer to account for market hours gaps (nights, weekends)
        self.minutes_lookback = int(self.candles * mins_per_candle * 1.4)

        self.start_datetime = payload.get('start_datetime', None)
        self.end_datetime   = payload.get('end_datetime', None)

        # Decision timestamp
        self.timestamp_mode     = payload.get('timestamp_mode', 'latest')
        self.decision_timestamp = payload.get('decision_timestamp', None)

        # ── Risk parameters ──────────────────────────────────────────
        self.base_sl_atr_multiple    = 1.5
        self.base_tp_atr_multiple    = 3.0
        self.partial_exit_1_ratio    = 1.5
        self.partial_exit_2_ratio    = 2.5

        # ── Decision tree thresholds ─────────────────────────────────
        self.adx_threshold               = 25.0
        self.rsi_oversold                = 30.0
        self.rsi_overbought              = 70.0
        self.roc_strong_threshold        = 2.0
        self.cci_threshold               = 100.0
        self.bb_width_expansion_factor   = 1.2
        self.volume_participation_factor = 1.5
        self.z_score_extreme_threshold   = 2.0

        # ── Indicator periods ────────────────────────────────────────
        self.ema_fast_period   = 9
        self.ema_slow_period   = 21
        self.macd_fast         = 8
        self.macd_slow         = 13
        self.macd_signal       = 21
        self.rsi_period        = 7
        self.roc_period        = 10
        self.cci_period        = 14
        self.adx_period        = 14
        self.bb_period         = 20
        self.bb_std            = 2
        self.atr_period        = 14
        self.volume_sma_period = 20
        self.z_score_period    = 20

        self.min_warmup_candles = 100

    def validate(self) -> tuple[bool, str]:
        """
        Validate the config. Returns (is_valid, error_message).
        The frontend validates too, but this is the server-side safety net.
        """
        if not self.api_key or not self.secret_key:
            return False, 'API key and secret key are required.'

        if not self.symbol:
            return False, 'Symbol is required.'

        if self.timeframe_str not in self.TIMEFRAME_MAP:
            return False, f'Invalid timeframe: {self.timeframe_str}'

        if self.range_mode == 'lookback':
            if self.candles < self.min_warmup_candles:
                return False, f'Lookback must be at least {self.min_warmup_candles} candles.'

        elif self.range_mode == 'daterange':
            if not self.start_datetime or not self.end_datetime:
                return False, 'Start and end datetime are required for date range mode.'

        if self.timestamp_mode == 'manual' and not self.decision_timestamp:
            return False, 'A decision timestamp is required when using manual mode.'

        return True, ''