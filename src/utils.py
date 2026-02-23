"""
Utility helpers for SYNAPSE web app.
Stripped down to only what the web responses need.
"""

import pandas as pd


def validate_sufficient_candles(df: pd.DataFrame, idx: int, min_candles: int = 100) -> tuple[bool, str]:
    """
    Confirm there are enough candles before idx for indicators to be meaningful.
    Returns (is_valid, error_message).
    """
    if idx < min_candles:
        return False, (
            f'The selected candle is only {idx} steps into the dataset. '
            f'At least {min_candles} candles are needed before it. '
            f'Please select a later timestamp or load more data.'
        )
    if idx >= len(df):
        return False, 'Selected index is outside the loaded dataset.'

    return True, ''


def format_timestamp(ts) -> str:
    """Return a clean UTC string from a pandas Timestamp."""
    if isinstance(ts, pd.Timestamp):
        return ts.strftime('%Y-%m-%d %H:%M:%S UTC')
    return str(ts)


def error_response(message: str) -> dict:
    """Standard error shape returned to the frontend."""
    return {'status': 'error', 'message': message}


def success_response(payload: dict) -> dict:
    """Standard success shape returned to the frontend."""
    return {'status': 'success', **payload}