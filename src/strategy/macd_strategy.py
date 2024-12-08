"""
MACD (Moving Average Convergence Divergence) Strategy

This strategy implements a technical indicator that uses the convergence and divergence
of moving averages to generate trading signals. It consists of three components:
    - MACD Line: Difference between fast and slow EMAs
    - Signal Line: EMA of the MACD line
    - Histogram: Difference between MACD and Signal lines

Key Components:
    - Fast EMA: More responsive to recent price changes
    - Slow EMA: Establishes the broader trend
    - Signal EMA: Smooths the MACD line
    
Trading Logic:
    - Buy (1): When MACD line crosses above Signal line (bullish signal)
    - Sell (-1): When MACD line crosses below Signal line (bearish signal)
    - Hold (0): When no crossing occurs

Parameters:
    - fast_period (int): Period for fast EMA (default: 12)
    - slow_period (int): Period for slow EMA (default: 26)
    - signal_period (int): Period for signal line EMA (default: 9)

Example:
    strategy = MACDStrategy({
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    })
    signals = strategy.generate_signals(price_data)

Notes:
    - Best suited for trending markets
    - May generate false signals in sideways markets
    - Consider using with volume or trend filters
    - Performance depends on chosen periods and market conditions
"""

from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the MACD Strategy.

        Args:
            parameters (Dict[str, Any], optional): Strategy parameters including:
                - fast_period: Period for fast EMA
                - slow_period: Period for slow EMA
                - signal_period: Period for signal line EMA
                If None, defaults will be used.

        Raises:
            ValueError: If fast_period >= slow_period
        """
        parameters = parameters or {}
        self.fast_period = parameters.get("fast_period", 12)
        self.slow_period = parameters.get("slow_period", 26)
        self.signal_period = parameters.get("signal_period", 9)
        self.validate_parameters()
        super().__init__(parameters)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicators.

        Args:
            data (pd.DataFrame): Historical price data with required columns:
                - close: Closing prices

        Returns:
            pd.DataFrame: Original data with additional columns:
                - MACD: MACD line
                - Signal: Signal line
                - Histogram: MACD histogram
        """
        data = data.copy()

        # Calculate EMAs
        fast_ema = data["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data["close"].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        data["MACD"] = fast_ema - slow_ema

        # Calculate Signal line
        data["Signal"] = data["MACD"].ewm(span=self.signal_period, adjust=False).mean()

        # Calculate Histogram
        data["Histogram"] = data["MACD"] - data["Signal"]

        return data

    def get_required_warmup_period(self) -> int:
        """
        Get the required warmup period for the strategy.
        For MACD, we need max(slow_period, fast_period + signal_period) periods
        """
        return max(self.slow_period, self.fast_period + self.signal_period)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD crossovers.

        The strategy generates:
            1 (Buy)  : When MACD crosses above Signal line
            -1 (Sell): When MACD crosses below Signal line
            0 (Hold) : When no crossing occurs

        Args:
            data (pd.DataFrame): Historical price data with required indicators

        Returns:
            pd.Series: Trading signals aligned with the input data's index
        """
        signals = pd.Series(0, index=data.index)

        # Generate signals using the clean data
        signals[data["MACD"] > data["Signal"]] = 1
        signals[data["MACD"] < data["Signal"]] = -1

        return signals

    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.

        Checks:
            1. Periods are positive integers
            2. Fast period is less than slow period
            3. Signal period is appropriate

        Returns:
            bool: True if parameters are valid

        Raises:
            ValueError: If parameters are invalid
        """
        if self.fast_period <= 0:
            raise ValueError("fast_period must be greater than 0")
        if self.slow_period <= 0:
            raise ValueError("slow_period must be greater than 0")
        if self.signal_period <= 0:
            raise ValueError("signal_period must be greater than 0")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        return True
