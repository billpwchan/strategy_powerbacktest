"""Ports of the TDX formula strategies BTSE and LEG.

The original TDX sources are not in the repository; these ports keep the logic of the
previous implementation, restated with TDX semantics, and document every assumption.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from powerbacktest.strategy.base import PlotSpec, Strategy, StrategyParams, as_bool, events
from powerbacktest.strategy.registry import register
from powerbacktest.strategy.tdx import CROSS, DMI, MA, REF, ZIG, point_in_time, zig_pivots


def _zig_cross_last(values: np.ndarray, pct: float, ma: int) -> float:
    """+1 if CROSS(ZIG, MA(ZIG, ma)) at the last bar of ``values``, -1 for the reverse, else 0.

    Only the last ``ma + 1`` ZIG values are needed, so they are interpolated directly.
    """
    n = len(values)
    if n < ma + 2:
        return np.nan
    pivots = zig_pivots(values, pct)
    tail = np.arange(n - ma - 1, n)
    zig = np.interp(tail, pivots, values[pivots])
    ma_now = zig[1:].mean()
    ma_prev = zig[:-1].mean()
    z_now, z_prev = zig[-1], zig[-2]
    if z_now > ma_now and z_prev <= ma_prev:
        return 1.0
    if ma_now > z_now and ma_prev <= z_prev:
        return -1.0
    return 0.0


class BTSEParams(StrategyParams):
    dmi_period: int = Field(25, ge=2, le=200, description="DMI N (old lookback_period)")
    adx_period: int = Field(15, ge=1, le=100, description="DMI M for ADX/ADXR")
    di_threshold: float = Field(25.0, ge=0, le=100)
    zig_pct: float = Field(10.0, gt=0, le=100, description="ZIG reversal threshold in percent")
    zig_ma: int = Field(2, ge=1, le=50)
    zig_mode: Literal["pit", "tdx"] = Field(
        "pit",
        description="pit: evaluate ZIG as seen at each bar (causal); tdx: TDX's repainting ZIG (future function)",
    )
    pit_window: int = Field(250, ge=30, le=5000)


@register
class BTSEStrategy(Strategy):
    """BTSE: DMI trend-strength filter plus a ZIG turn.

    STR1..STR9 in the old port are TDX's DMI: STR6 = PDI, STR7 = MDI, STR8 = ADX,
    STR9 = ADXR. The rules are

    * A: MDI > PDI and MDI > 25 and PDI < 25 (a strong down-move)
    * buy:  A and CROSS(ZIG(3, N), MA(ZIG, 2))  -- ZIG turns up
    * sell: CROSS(MA(ZIG, 2), ZIG)              -- ZIG turns down

    TDX's ZIG repaints, so TDX's own backtest of this formula uses future data. The default
    ``zig_mode: pit`` recomputes ZIG on each bar's history and uses only its latest leg, which
    is what the formula would actually have shown when the bar closed.
    """

    name = "btse"
    title = "BTSE (DMI + ZIG)"
    description = "Enter on a ZIG up-turn after a strong DMI down-move; exit on a ZIG down-turn."
    Params = BTSEParams
    plots = (
        PlotSpec("pdi", "lower", "line", "PDI"),
        PlotSpec("mdi", "lower", "line", "MDI"),
        PlotSpec("adx", "lower", "line", "ADX"),
    )
    params: BTSEParams

    def warmup_bars(self) -> int:
        p = self.params
        return p.dmi_period + 2 * p.adx_period + 1

    @property
    def lookahead_safe(self) -> bool:
        return self.params.zig_mode == "pit"

    def warnings(self) -> list[str]:
        if self.params.zig_mode == "tdx":
            return [
                "BTSE runs with zig_mode=tdx: ZIG is a future function, so these results use "
                "information not available at trade time and overstate real performance."
            ]
        return []

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        dmi = DMI(bars["high"], bars["low"], bars["close"], p.dmi_period, p.adx_period)
        close = bars["close"]
        if p.zig_mode == "tdx":
            zig = ZIG(close, p.zig_pct)
            zig_ma = MA(zig, p.zig_ma)
            turn = pd.Series(0.0, index=bars.index)
            turn[CROSS(zig, zig_ma)] = 1.0
            turn[CROSS(zig_ma, zig)] = -1.0
            dmi["zig"] = zig
        else:
            turn = point_in_time(
                lambda w: pd.Series(
                    [_zig_cross_last(w["close"].to_numpy(float), p.zig_pct, p.zig_ma)]
                ),
                bars,
                p.pit_window,
                min_bars=p.zig_ma + 2,
            )
        dmi["zig_turn"] = turn
        return dmi

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        p = self.params
        a = (
            (ind["mdi"] > ind["pdi"])
            & (ind["mdi"] > p.di_threshold)
            & (ind["pdi"] < p.di_threshold)
        )
        return events(a & (ind["zig_turn"] == 1.0), ind["zig_turn"] == -1.0)


class LEGParams(StrategyParams):
    pass


@register
class LEGStrategy(Strategy):
    """LEG: two-bar reversal pattern.

    * VAR1: CLOSE > REF(CLOSE, 1) and CLOSE > REF(CLOSE, 2)
    * VARD: CLOSE < REF(CLOSE, 1) and CLOSE < REF(CLOSE, 2)
    * buy (ENTER): VAR1 and REF(VARD, 1); sell (LEAVE): VARD and REF(VAR1, 1)

    The old port also computed MA, KDJ, MACD and Fibonacci levels but never used them in its
    signals; without the original formula it is unknown how they combined, so they are not
    reproduced here.
    """

    name = "leg"
    title = "LEG reversal pattern"
    description = "Buy on an up-reversal bar after a down bar, sell on the mirror pattern."
    Params = LEGParams

    def warmup_bars(self) -> int:
        return 3

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        c = bars["close"]
        up = (c > REF(c, 1)) & (c > REF(c, 2))
        down = (c < REF(c, 1)) & (c < REF(c, 2))
        return pd.DataFrame(
            {"up_bar": up.astype(float), "down_bar": down.astype(float)}, index=bars.index
        )

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        up, down = ind["up_bar"] == 1.0, ind["down_bar"] == 1.0
        return events(
            up & as_bool(REF(down, 1)),
            down & as_bool(REF(up, 1)),
        )
