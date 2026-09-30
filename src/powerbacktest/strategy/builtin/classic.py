"""Classic indicator strategies: MACD, moving-average cross, KDJ and RSI.

MACD, KDJ and RSI reproduce the buy/sell rules of the live strategies in futu_algo
(``MACD_Cross``, ``KDJ_Cross``, ``RSI_Threshold``), computed with TDX formulas, so a
backtest here tests the same rules the live bot trades.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from powerbacktest.strategy.base import PlotSpec, Strategy, StrategyParams, events, state
from powerbacktest.strategy.registry import register
from powerbacktest.strategy.tdx import CROSS, EMA, KDJ, MA, MACD, REF, RSI

Mode = Literal["cross", "state"]


class MACDParams(StrategyParams):
    fast_period: int = Field(12, ge=2, le=200)
    slow_period: int = Field(26, ge=3, le=400)
    signal_period: int = Field(9, ge=2, le=200)
    mode: Mode = Field(
        "cross", description="cross: act on DIF/DEA crossings; state: long while DIF > DEA"
    )

    @model_validator(mode="after")
    def _order(self) -> MACDParams:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        return self


@register
class MACDStrategy(Strategy):
    name = "macd"
    title = "MACD cross"
    description = "Buy when DIF crosses above DEA, sell when it crosses below (TDX MACD)."
    Params = MACDParams
    plots = (
        PlotSpec("dif", "lower", "line", "DIF"),
        PlotSpec("dea", "lower", "line", "DEA"),
        PlotSpec("macd", "lower", "histogram", "MACD"),
    )
    params: MACDParams

    def warmup_bars(self) -> int:
        # EMAs are seeded with the first bar; ~3x the longest span removes the seed's effect.
        return 3 * (self.params.slow_period + self.params.signal_period)

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return MACD(bars["close"], p.fast_period, p.slow_period, p.signal_period)

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        if self.params.mode == "state":
            return state(ind["dif"] > ind["dea"])
        return events(CROSS(ind["dif"], ind["dea"]), CROSS(ind["dea"], ind["dif"]))


class MACrossParams(StrategyParams):
    short_window: int = Field(20, ge=1, le=500)
    long_window: int = Field(50, ge=2, le=1000)
    ma_type: Literal["sma", "ema"] = "sma"
    mode: Mode = "cross"

    @model_validator(mode="after")
    def _order(self) -> MACrossParams:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        return self


@register
class MACrossStrategy(Strategy):
    name = "ma_cross"
    title = "Moving-average cross"
    description = "Buy when the short MA crosses above the long MA, sell on the reverse cross."
    Params = MACrossParams
    plots = (
        PlotSpec("ma_short", "price", "line", "MA short"),
        PlotSpec("ma_long", "price", "line", "MA long"),
    )
    params: MACrossParams

    def warmup_bars(self) -> int:
        return self.params.long_window * (3 if self.params.ma_type == "ema" else 1)

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        fn = MA if p.ma_type == "sma" else EMA
        return pd.DataFrame(
            {
                "ma_short": fn(bars["close"], p.short_window),
                "ma_long": fn(bars["close"], p.long_window),
            },
            index=bars.index,
        )

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        s, lg = ind["ma_short"], ind["ma_long"]
        if self.params.mode == "state":
            return state((s > lg).where(lg.notna()))
        return events(CROSS(s, lg), CROSS(lg, s))


class KDJParams(StrategyParams):
    n: int = Field(9, ge=2, le=200, description="RSV lookback (futu_algo fast_k)")
    m1: int = Field(3, ge=1, le=50, description="K smoothing (slow_k)")
    m2: int = Field(3, ge=1, le=50, description="D smoothing (slow_d)")
    over_buy: float = Field(80.0, ge=50, le=100)
    over_sell: float = Field(20.0, ge=0, le=50)


@register
class KDJStrategy(Strategy):
    name = "kdj"
    title = "KDJ oversold/overbought cross"
    description = (
        "futu_algo KDJ_Cross rules: buy on a K-over-D turn while D is below the oversold "
        "level, sell on a K-under-D turn while D is above the overbought level."
    )
    Params = KDJParams
    plots = (
        PlotSpec("k", "lower", "line", "K"),
        PlotSpec("d", "lower", "line", "D"),
        PlotSpec("j", "lower", "line", "J"),
    )
    params: KDJParams

    def warmup_bars(self) -> int:
        return self.params.n + 10 * max(self.params.m1, self.params.m2)

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return KDJ(bars["high"], bars["low"], bars["close"], p.n, p.m1, p.m2)

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        p = self.params
        k, d = ind["k"], ind["d"]
        pk, pd_ = REF(k, 1), REF(d, 1)
        buy = (p.over_sell > d) & (d > pd_) & (pd_ > pk) & (k > pk) & (k > d)
        sell = (p.over_buy < d) & (d < pd_) & (pd_ < pk) & (k < pk) & (k < d)
        return events(buy, sell)


class RSIParams(StrategyParams):
    period: int = Field(6, ge=2, le=200)
    lower: float = Field(30.0, ge=0, le=100)
    upper: float = Field(70.0, ge=0, le=100)

    @model_validator(mode="after")
    def _order(self) -> RSIParams:
        if self.lower >= self.upper:
            raise ValueError("lower must be below upper")
        return self


@register
class RSIStrategy(Strategy):
    name = "rsi"
    title = "RSI threshold"
    description = (
        "futu_algo RSI_Threshold rules: buy when RSI drops through the lower level, sell when "
        "it rises through the upper level (TDX RSI)."
    )
    Params = RSIParams
    plots = (PlotSpec("rsi", "lower", "line", "RSI"),)
    params: RSIParams

    def warmup_bars(self) -> int:
        return 10 * self.params.period

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"rsi": RSI(bars["close"], self.params.period)}, index=bars.index)

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        p = self.params
        rsi, prev = ind["rsi"], REF(ind["rsi"], 1)
        return events((rsi < p.lower) & (prev > p.lower), (rsi > p.upper) & (prev < p.upper))
