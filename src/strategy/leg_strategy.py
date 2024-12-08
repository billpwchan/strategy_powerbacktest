from typing import Dict
import numpy as np
import pandas as pd
import talib
from .base_strategy import BaseStrategy


class LEGStrategy(BaseStrategy):
    """
    LEG (Level Entry Guide) Strategy
    Combines multiple technical indicators including MA, RSI, MACD, and price patterns
    """

    def __init__(self, parameters: Dict = None):
        super().__init__(parameters)
        self.ma_period = parameters.get("ma_period", 17)
        self.vol_period = parameters.get("vol_period", 88)
        self.rsi_period = parameters.get("rsi_period", 9)
        self.pattern_lookback = parameters.get("pattern_lookback", 120)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators used in the strategy"""
        close, high, low = data["close"], data["high"], data["low"]

        # CG: MA(C,17)
        cg = talib.MA(close, timeperiod=self.ma_period)
        fl = talib.MAX(cg, timeperiod=2)
        fs = cg - (fl - cg)

        # Volume indicators
        amv0 = data["volume"] * (data["open"] + close) / 2
        xx01 = talib.SUM(amv0, timeperiod=self.vol_period) / talib.SUM(
            data["volume"], timeperiod=self.vol_period
        )

        # Stochastic
        rsv = (
            (close - talib.MIN(low, timeperiod=9))
            / (talib.MAX(high, timeperiod=9) - talib.MIN(low, timeperiod=9))
            * 100
        )
        k = talib.SMA(rsv, timeperiod=3)
        d = talib.SMA(k, timeperiod=3)

        # MACD
        macd, signal, hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Fibonacci levels
        high_1 = talib.MAX(high, timeperiod=self.pattern_lookback).shift(3)
        low_1 = talib.MIN(low, timeperiod=self.pattern_lookback).shift(3)
        range_1 = high_1 - low_1

        h1 = high_1 - range_1 * 0.191
        h2 = high_1 - range_1 * 0.382
        h3 = high_1 - range_1 * 0.5
        h4 = high_1 - range_1 * 0.618
        h5 = high_1 - range_1 * 0.809

        return pd.DataFrame(
            {
                "cg": cg,
                "fl": fl,
                "fs": fs,
                "xx01": xx01,
                "k": k,
                "d": d,
                "macd": macd,
                "signal": signal,
                "h1": h1,
                "h5": h5,
                "high_1": high_1,
                "low_1": low_1,
            }
        )

    def check_pattern(self, data: pd.DataFrame) -> tuple:
        """Check for specific price patterns"""
        close = data["close"]
        prev_close = close.shift(1)
        prev_close2 = close.shift(2)

        var1 = (close > prev_close) & (close > prev_close2)
        vard = (close < prev_close) & (close < prev_close2)

        # Pattern continuation checks
        var19 = var1 & vard.shift(1)  # Buy signal
        var1a = vard & var1.shift(1)  # Sell signal

        return var19, var1a

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on the strategy rules"""
        indicators = self.calculate_indicators(data)
        enter_pattern, leave_pattern = self.check_pattern(data)

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Buy signals (ENTER conditions)
        buy_signals = enter_pattern

        # Sell signals (LEAVE conditions)
        sell_signals = leave_pattern

        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals
