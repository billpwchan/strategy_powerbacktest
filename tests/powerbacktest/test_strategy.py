from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest
from helpers import make_daily_bars

from powerbacktest.errors import StrategyError
from powerbacktest.strategy import available, check_lookahead, create_strategy, get_strategy_class
from powerbacktest.strategy.base import PlotSpec, Strategy, StrategyParams, events
from powerbacktest.strategy.registry import load_strategy_paths
from powerbacktest.strategy.tdx import (
    BARSLAST,
    CROSS,
    DMI,
    EMA,
    HHV,
    KDJ,
    LLV,
    MA,
    MACD,
    REF,
    RSI,
    SMA,
    ZIG,
    point_in_time,
    zig_pivots,
)

BARS = make_daily_bars(n=400, seed=11, vol=0.025)


# ------------------------------------------------------------------ tdx


def test_ma_is_undefined_until_n_bars():
    x = pd.Series([1.0, 2, 3, 4, 5])
    assert MA(x, 3).tolist()[:2] == [pytest.approx(np.nan, nan_ok=True)] * 2
    assert MA(x, 3).iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_sma_is_tdx_recursive_not_rolling():
    x = pd.Series([10.0, 20.0, 30.0])
    # Y1 = X1; Y2 = (1*20 + 2*10)/3; Y3 = (1*30 + 2*Y2)/3
    y2 = (20 + 2 * 10) / 3
    y3 = (30 + 2 * y2) / 3
    assert SMA(x, 3, 1).tolist() == pytest.approx([10.0, y2, y3])
    with pytest.raises(ValueError):
        SMA(x, 3, 4)


def test_ema_matches_recursive_definition():
    x = pd.Series([1.0, 2.0, 4.0])
    a = 2 / (3 + 1)
    e2 = a * 2 + (1 - a) * 1
    assert EMA(x, 3).tolist() == pytest.approx([1.0, e2, a * 4 + (1 - a) * e2])


def test_cross_hhv_llv_barslast():
    a = pd.Series([1.0, 2, 3, 2, 1])
    b = pd.Series([2.0, 2, 2, 2, 2])
    assert CROSS(a, b).tolist() == [False, False, True, False, False]
    assert CROSS(b, a).tolist() == [False, False, False, False, True]
    assert HHV(a, 2).tolist()[1:] == [2, 3, 3, 2]
    assert LLV(a, 0).tolist() == [1, 1, 1, 1, 1]
    assert BARSLAST(pd.Series([False, True, False, False, True])).tolist()[1:] == [0, 1, 2, 0]


def test_macd_histogram_is_doubled_like_tdx_and_futu_algo():
    c = BARS["close"]
    m = MACD(c, 12, 26, 9)
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    np.testing.assert_allclose(m["macd"], 2 * (dif - dea))


def test_kdj_uses_recursive_smoothing():
    k = KDJ(BARS["high"], BARS["low"], BARS["close"], 9, 3, 3)
    rsv = (
        (BARS["close"] - BARS["low"].rolling(9).min())
        / (BARS["high"].rolling(9).max() - BARS["low"].rolling(9).min())
        * 100
    )
    expected_k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    np.testing.assert_allclose(k["k"].dropna(), expected_k.dropna())
    assert ((k["k"].dropna() >= 0) & (k["k"].dropna() <= 100)).all()


def test_rsi_bounds_and_dmi_shape():
    r = RSI(BARS["close"], 6).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    d = DMI(BARS["high"], BARS["low"], BARS["close"], 14, 6)
    assert list(d.columns) == ["pdi", "mdi", "adx", "adxr"]
    assert d.dropna().shape[0] > 300


def test_zig_pivots_and_interpolation():
    x = pd.Series([100.0, 105, 112, 108, 99, 95, 101, 110])
    piv = zig_pivots(x.to_numpy(), 10)
    assert piv[0] == 0 and piv[-1] == len(x) - 1
    assert 2 in piv and 5 in piv  # 112 peak, 95 trough
    z = ZIG(x, 10)
    assert z.iloc[2] == 112 and z.iloc[5] == 95
    assert z.iloc[3] == pytest.approx(112 + (95 - 112) / 3)


def test_point_in_time_is_causal():
    frame = BARS.iloc[:120]
    pit = point_in_time(lambda w: ZIG(w["close"], 5), frame, window=60)
    # Evaluated at its own last bar, ZIG always equals the close.
    np.testing.assert_allclose(pit.iloc[1:], frame["close"].iloc[1:])


# ----------------------------------------------------------- strategies


@pytest.mark.parametrize("name", ["macd", "ma_cross", "kdj", "rsi", "btse", "leg"])
def test_builtin_strategies_are_causal_and_well_formed(name):
    strategy = create_strategy(name)
    ind, sig = strategy.run(BARS)
    assert sig.index.equals(BARS.index)
    assert set(sig.dropna().unique()) <= {0.0, 1.0}
    report = check_lookahead(strategy, BARS, checkpoints=6)
    assert report.ok, report.summary()
    for plot in strategy.plots:
        assert plot.column in ind.columns


def test_state_modes_emit_every_bar():
    macd_state = create_strategy("macd", {"mode": "state"})
    _, sig = macd_state.run(BARS)
    assert sig.notna().all()
    ma_state = create_strategy("ma_cross", {"mode": "state", "short_window": 5, "long_window": 20})
    _, sig = ma_state.run(BARS)
    assert sig.iloc[:19].isna().all() and sig.iloc[19:].notna().all()


def test_btse_tdx_mode_is_flagged_as_lookahead():
    strategy = create_strategy("btse", {"zig_mode": "tdx", "zig_pct": 5})
    assert not strategy.lookahead_safe and strategy.warnings()
    report = check_lookahead(strategy, BARS, checkpoints=8)
    assert not report.ok
    assert "look-ahead detected" in report.summary()


def test_macd_cross_matches_futu_algo_rule():
    strategy = create_strategy("macd")
    ind, sig = strategy.run(BARS)
    dif, dea = ind["dif"], ind["dea"]
    buy = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    sell = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    assert (sig == 1.0).equals(buy & ~sell)
    assert (sig == 0.0).sum() == (sell & ~buy).sum()


def test_leg_pattern():
    closes = [10, 9, 8, 9.5, 10, 9, 8.5]
    bars = make_daily_bars(n=len(closes))
    bars["close"] = closes
    _, sig = create_strategy("leg").run(bars)
    # bar 3: 9.5 > 8 and > 9 (up), bar 2 was down -> buy; bar 5: 9 < 10, 9 < 9.5 (down), bar 4 up -> sell
    assert sig.iloc[3] == 1.0 and sig.iloc[5] == 0.0
    assert sig.drop(sig.index[[3, 5]]).isna().all()


def test_param_validation_and_decide_last():
    with pytest.raises(StrategyError):
        create_strategy("macd", {"fast_period": 30, "slow_period": 20})
    with pytest.raises(StrategyError):
        create_strategy("macd", {"unknown": 1})
    with pytest.raises(StrategyError):
        create_strategy("nope")
    strategy = create_strategy("macd")
    _, sig = strategy.run(BARS)
    for end in (150, 260, 399):
        assert strategy.decide_last(BARS.iloc[: end + 1]) == pytest.approx(
            sig.iloc[end], nan_ok=True
        )


def test_registry_lists_builtins_with_schema():
    names = set(available())
    assert {"macd", "ma_cross", "kdj", "rsi", "btse", "leg"} <= names
    schema = get_strategy_class("rsi").param_schema()
    assert "period" in schema["properties"]


def test_user_strategy_from_file(tmp_path):
    path = tmp_path / "my_strat.py"
    path.write_text(
        textwrap.dedent(
            """
            import pandas as pd
            from powerbacktest.strategy import Strategy, StrategyParams, state

            class P(StrategyParams):
                level: float = 100.0

            class AboveLevel(Strategy):
                name = "above_level_test"
                Params = P

                def indicators(self, bars):
                    return pd.DataFrame(index=bars.index)

                def signals(self, bars, ind):
                    return state(bars["close"] > self.params.level)
            """
        )
    )
    cls = get_strategy_class(f"{path}:AboveLevel")
    assert cls.name == "above_level_test"
    load_strategy_paths([tmp_path])
    assert "above_level_test" in available()


def test_invalid_signal_values_are_rejected():
    class Bad(Strategy):
        name = ""
        Params = StrategyParams
        plots = (PlotSpec("x"),)

        def indicators(self, bars):
            return pd.DataFrame({"x": 1.0}, index=bars.index)

        def signals(self, bars, ind):
            return pd.Series(2.0, index=bars.index)

    with pytest.raises(StrategyError):
        Bad().run(BARS)
    assert events(pd.Series([True, True]), pd.Series([False, True])).tolist()[0] == 1.0
    assert np.isnan(events(pd.Series([True, True]), pd.Series([False, True])).tolist()[1])
    _ = REF
