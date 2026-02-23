"""
Risk management module for SYNAPSE Scalping Strategy.

This module handles all risk management calculations including stop loss,
take profit, partial exits, and position sizing.
"""

import pandas as pd
from typing import Dict

from .logger import get_logger
from .config import Config

logger = get_logger()


class RiskManager:
    """Manages risk parameters for trading decisions."""

    def __init__(self, config: Config):
        """
        Initialize RiskManager with configuration.

        Parameters:
            config: Configuration object containing risk parameters
        """
        self.config = config
        logger.info("RiskManager initialized")

    def calculate_risk_parameters(self, decision: Dict) -> Dict:
        """
        Calculate risk management parameters for a trade signal.

        Parameters:
            decision: Dictionary containing trade decision info

        Returns:
            Dict: Dictionary with added risk management parameters
        """
        if decision['decision'] != 'TRADE':
            logger.debug("No trade decision - skipping risk calculations")
            return decision

        logger.info(f"Calculating risk parameters for {decision['direction']} trade")

        close_val = decision['close']
        atr_val = decision['atr']
        z_score_val = decision['z_score']
        signal_value = decision['signal']

        # Adjust based on z-score (statistical edge)
        z_factor = abs(z_score_val) if not pd.isna(z_score_val) else 0
        z_factor = min(z_factor, 2.0)  # Cap at 2

        logger.debug(f"Z-Factor: {z_factor:.2f}")

        if signal_value == 1:  # LONG
            sl_distance, tp_distance = self._calculate_long_risk(
                atr_val,
                z_factor
            )
            decision.update(self._create_long_risk_params(
                close_val,
                sl_distance,
                tp_distance
            ))
        else:  # SHORT
            sl_distance, tp_distance = self._calculate_short_risk(
                atr_val,
                z_factor
            )
            decision.update(self._create_short_risk_params(
                close_val,
                sl_distance,
                tp_distance
            ))

        # Add common risk parameters
        decision['risk_amount'] = sl_distance
        decision['reward_amount'] = tp_distance
        decision['risk_reward_ratio'] = (
            tp_distance / sl_distance if sl_distance > 0 else 0
        )

        logger.info(f"Risk parameters calculated - R:R = {decision['risk_reward_ratio']:.2f}:1")

        return decision

    def _calculate_long_risk(self, atr: float, z_factor: float) -> tuple:
        """Calculate stop loss and take profit distances for LONG trade."""
        sl_distance = (atr * self.config.base_sl_atr_multiple *
                       (1 - 0.3 * z_factor))
        tp_distance = (atr * self.config.base_tp_atr_multiple *
                       (1 + 0.5 * z_factor))
        return sl_distance, tp_distance

    def _calculate_short_risk(self, atr: float, z_factor: float) -> tuple:
        """Calculate stop loss and take profit distances for SHORT trade."""
        sl_distance = (atr * self.config.base_sl_atr_multiple *
                       (1 + 0.3 * z_factor))
        tp_distance = (atr * self.config.base_tp_atr_multiple *
                       (1 - 0.5 * z_factor))
        return sl_distance, tp_distance

    def _create_long_risk_params(
            self,
            close: float,
            sl_distance: float,
            tp_distance: float
    ) -> Dict:
        """Create risk parameter dictionary for LONG trade."""
        return {
            'stop_loss': close - sl_distance,
            'take_profit': close + tp_distance,
            'partial_exit_1': close + (sl_distance * self.config.partial_exit_1_ratio),
            'partial_exit_2': close + (sl_distance * self.config.partial_exit_2_ratio),
            'trailing_stop': close  # Starts at break-even
        }

    def _create_short_risk_params(
            self,
            close: float,
            sl_distance: float,
            tp_distance: float
    ) -> Dict:
        """Create risk parameter dictionary for SHORT trade."""
        return {
            'stop_loss': close + sl_distance,
            'take_profit': close - tp_distance,
            'partial_exit_1': close - (sl_distance * self.config.partial_exit_1_ratio),
            'partial_exit_2': close - (sl_distance * self.config.partial_exit_2_ratio),
            'trailing_stop': close  # Starts at break-even
        }