from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import numpy as np
import pandas as pd

from src.config import Config
from src.data_loader import DataLoader
from src.logger import setup_logger
from src.utils import error_response, success_response
from src.visualization import build_chart
from src.indicators import IndicatorCalculator
from src.decision_engine import DecisionEngine
from src.risk_manager import RiskManager

app = Flask(__name__)
CORS(app)
logger = setup_logger()

# In-memory cache — keyed by session_id
# Stores the fully calculated DataFrame so /api/decision never re-fetches
_cache = {}


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/chart', methods=['POST'])
def chart():
    """
    Heavy endpoint — fetches data, calculates all indicators,
    builds chart, runs initial decision. Stores df in cache.
    """
    payload = request.get_json()
    if not payload:
        return jsonify(error_response('No payload received.')), 400

    # 1. Config + validate
    config = Config(payload)
    valid, msg = config.validate()
    if not valid:
        return jsonify(error_response(msg)), 400

    # 2. Fetch
    loader = DataLoader(config.api_key, config.secret_key)
    try:
        df = loader.fetch(config)
    except Exception as e:
        return jsonify(error_response(f'Data fetch failed: {str(e)}')), 500

    valid, msg = loader.validate(df, config)
    if not valid:
        return jsonify(error_response(msg)), 400

    # 3. Decision index
    try:
        decision_idx = loader.get_decision_index(df, config)
    except ValueError as e:
        return jsonify(error_response(str(e))), 400

    # 4. Indicators
    calc = IndicatorCalculator(config)
    df   = calc.calculate(df)

    # 5. VWAP
    df = _add_vwap(df)

    # 6. Cache the df — generate a session id to return to frontend
    session_id = str(uuid.uuid4())
    _cache[session_id] = {
        'df':     df,
        'config': config
    }
    # Prune old sessions if cache grows too large
    if len(_cache) > 20:
        oldest = next(iter(_cache))
        del _cache[oldest]

    # 7. Run initial decision
    decision = _run_decision(df, config, decision_idx)

    # 8. Build chart
    fig_dict = build_chart(df, config.symbol, decision_idx)

    return jsonify(success_response({
        'figure':             fig_dict,
        'candle_count':       len(df),
        'decision_timestamp': str(df.index[decision_idx]),
        'decision_idx':       decision_idx,
        'symbol':             config.symbol,
        'decision':           decision,
        'session_id':         session_id,
    }))


@app.route('/api/decision', methods=['POST'])
def decision():
    """
    Lightweight endpoint — reruns decision engine at a new candle index.
    Uses cached df, no re-fetch, no indicator recalculation.
    """
    payload = request.get_json()
    if not payload:
        return jsonify(error_response('No payload received.')), 400

    session_id   = payload.get('session_id')
    decision_idx = payload.get('decision_idx')

    if not session_id or session_id not in _cache:
        return jsonify(error_response(
            'Session expired or not found. Please run a full analysis first.'
        )), 400

    if decision_idx is None:
        return jsonify(error_response('No decision index provided.')), 400

    cached = _cache[session_id]
    df     = cached['df']
    config = cached['config']

    decision_idx = int(decision_idx)

    # Validate index has enough warmup candles before it
    valid, msg = _validate_decision_idx(df, decision_idx, config)
    if not valid:
        return jsonify(error_response(msg)), 400

    # Run decision at new index
    decision = _run_decision(df, config, decision_idx)

    # Build updated vertical line positions for chart
    decision_timestamp = str(df.index[decision_idx])

    return jsonify(success_response({
        'decision':           decision,
        'decision_idx':       decision_idx,
        'decision_timestamp': decision_timestamp,
    }))


# ------------------------------------------------------------------ #
# SHARED HELPERS
# ------------------------------------------------------------------ #

def _run_decision(df: pd.DataFrame, config: Config, idx: int) -> dict:
    """Run decision engine + risk manager at a given candle index."""
    candle   = df.iloc[idx]
    prev_obv = df['obv'].iloc[idx - 1] if idx > 0 else None

    engine   = DecisionEngine(config)
    decision = engine.make_decision(candle, prev_obv)

    if decision['decision'] == 'TRADE':
        risk_mgr = RiskManager(config)
        decision = risk_mgr.calculate_risk_parameters(decision)

    return decision


def _validate_decision_idx(df: pd.DataFrame, idx: int, config) -> tuple[bool, str]:
    """Check that idx is valid and has enough warmup candles before it."""
    if idx < 0 or idx >= len(df):
        return False, f'Candle index {idx} is out of range (0–{len(df) - 1}).'

    if idx < config.min_warmup_candles:
        return False, (
            f'Only {idx} candles exist before the selected candle. '
            f'At least {config.min_warmup_candles} are needed for accurate indicators. '
            f'Please select a later candle.'
        )

    return True, ''


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['tp']   = (df['high'] + df['low'] + df['close']) / 3
    df['date'] = df.index.normalize()

    vwap_values = []
    for date, group in df.groupby('date'):
        cum_tp_vol = (group['tp'] * group['volume']).cumsum()
        cum_vol    = group['volume'].cumsum()
        vwap_values.append(cum_tp_vol / cum_vol)

    df['vwap'] = pd.concat(vwap_values)
    df.drop(columns=['tp', 'date'], inplace=True)
    return df


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)