"""Regression tests for defects found in review of the 1.0 engine."""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from helpers import bars_from_prices, make_daily_bars
from test_engine import FREE, NO_SLIP, Fixed, simulate

from powerbacktest.config import ExecutionConfig, PortfolioConfig, RiskConfig, RunConfig
from powerbacktest.engine.runner import COMPOSITE, run_on_bars
from powerbacktest.engine.simulator import Simulator, SymbolFeed, fit_lots
from powerbacktest.market.costs import HKCostModel, SimpleCostModel, USCostModel
from powerbacktest.market.instrument import Instrument
from powerbacktest.optimize import optimize
from powerbacktest.strategy import BarContext, create_strategy
from powerbacktest.strategy.base import Strategy, StrategyParams

EX = ExecutionConfig(slippage_ticks=0, entry_requires_fresh_signal=False)


def test_gap_through_take_profit_fills_at_open_not_at_stop():
    # entry 100 at bar 1; bar 2 lifts the peak to 108 (trail 102.6); bar 3 gaps to 111
    bars = bars_from_prices(
        [100, 100, 104, 111],
        [100, 100, 107, 111],
        highs=[100, 101, 108, 112],
        lows=[99, 99, 103, 102],
    )
    book = simulate(
        bars, [1], ex=EX, risk=RiskConfig(take_profit=0.10, trailing_stop=0.05), capital=100_000
    )
    sell = book.fills[1]
    assert (sell.bar_index, sell.price, sell.reason) == (3, 111.0, "take_profit")


def test_open_trade_pnl_counts_partial_exits():
    bars = bars_from_prices([10, 10, 14, 12, 12], [10, 10, 14, 12, 12])
    bars["volume"] = [1e9, 1e9, 1000, 1000, 1000]
    ex = ExecutionConfig(slippage_ticks=0, max_volume_pct=0.1, entry_requires_fresh_signal=False)
    book = simulate(bars, [1, None, 0], ex=ex, costs=SimpleCostModel(0.001), capital=3_010.0)
    trade = book.trades[0]
    assert trade.is_open and 0 < trade.open_quantity < trade.quantity
    assert trade.pnl == pytest.approx(book.equity.iloc[-1] - 3_010.0)


def test_max_positions_frees_slot_of_position_being_sold():
    a, b = Instrument("HK.00001", lot_size=100), Instrument("HK.00002", lot_size=100)
    bars = bars_from_prices([10.0] * 6)
    feeds = []
    for inst, sig in ((a, [None, 1, None, 0]), (b, [None, None, None, 1])):
        strat = Fixed(signals=tuple(sig))
        ind, s = strat.run(bars)
        feeds.append(SymbolFeed(inst, bars, ind, s.to_numpy(float), 0, len(bars) - 1))
    book = Simulator(
        feeds,
        Fixed(),
        FREE,
        capital=10_000.0,
        execution=NO_SLIP,
        risk=RiskConfig(),
        portfolio=PortfolioConfig(max_positions=1),
        sizer="all_in",
        name="p",
    ).run()
    assert [(f.symbol, f.side, f.bar_index) for f in book.fills] == [
        ("HK.00001", "BUY", 2),
        ("HK.00001", "SELL", 4),
        ("HK.00002", "BUY", 4),
    ]


def test_excursions_only_cover_bars_held():
    bars = bars_from_prices([10, 10, 10], [10, 10, 10], highs=[10, 10, 20], lows=[10, 10, 5])
    book = simulate(bars, [1, 0], ex=EX)  # buy bar 1 open, sell bar 2 open
    trade = book.trades[0]
    assert trade.mae_pct == pytest.approx(0.0) and trade.mfe_pct == pytest.approx(0.0)


def test_nan_volume_with_volume_cap_does_not_crash():
    bars = bars_from_prices([10.0] * 5)
    bars["volume"] = [np.nan, 1e6, np.nan, 1e6, 1e6]
    ex = ExecutionConfig(slippage_ticks=0, max_volume_pct=0.1, entry_requires_fresh_signal=False)
    book = simulate(bars, [None, 1], ex=ex)  # order for bar 2, whose volume is missing
    assert book.fills == [] and book.event_counts == {"buy_skipped": 1}


def test_capped_slices_never_sell_below_their_fees():
    penny = Instrument("HK.08888", lot_size=1000)
    bars = bars_from_prices([0.012] * 8)
    bars["volume"] = [1e12] * 3 + [20_000] * 5  # after entry, 5% of volume is a single lot
    ex = ExecutionConfig(slippage_ticks=0, max_volume_pct=0.05, entry_requires_fresh_signal=False)
    book = simulate(bars, [1, None, 0], inst=penny, costs=HKCostModel(), capital=100.0, ex=ex)
    assert [f.side for f in book.fills] == ["BUY"]
    assert book.event_counts.get("sell_deferred", 0) >= 4
    assert (book.cash >= 0).all() and book.trades[0].is_open


def test_fit_lots_is_fast_and_exact():
    aapl = Instrument("US.XYZ")
    model = USCostModel()
    t0 = time.perf_counter()
    qty, fees = fit_lots(
        2_000_000,
        1,
        0.05,
        lambda q: model.fees("BUY", q, 0.05, aapl, pd.Timestamp("2024-06-03").date()),
        100_000.0,
    )
    assert time.perf_counter() - t0 < 0.1
    assert qty * 0.05 + fees.total <= 100_000.0
    assert (qty + 1) * 0.05 + model.fees(
        "BUY", qty + 1, 0.05, aapl, pd.Timestamp("2024-06-03").date()
    ).total > 100_000.0


def _cfg(**bt):
    return RunConfig.model_validate(
        {"backtest": {"strategy": "macd", "start": "2023-01-01", "end": "2024-06-30", **bt}}
    )


def test_index_benchmark_is_rebased_to_each_books_span():
    late = make_daily_bars(n=200, start="2023-09-01", seed=4, price=50)  # starts trading ~Nov 2023+
    index_bars = make_daily_bars(n=700, start="2022-01-03", seed=9, price=20_000)
    index_bars.loc[index_bars.index < "2024-01-01", ["open", "high", "low", "close"]] *= 0.5
    hsi = Instrument("HK.800000", lot_size=1, security_type="IDX", name="HSI")
    cfg = _cfg(symbols=["HK.02000"])
    result = run_on_bars(
        cfg,
        create_strategy("macd"),
        {"HK.02000": late},
        {"HK.02000": Instrument("HK.02000", lot_size=100)},
        benchmark=(index_bars, hsi),
    )
    book = result.books["HK.02000"]
    assert book.benchmark.iloc[0] == pytest.approx(book.initial_capital)
    span = index_bars.loc[
        (index_bars.index >= book.equity.index[0]) & (index_bars.index <= book.equity.index[-1]),
        "close",
    ]
    assert book.metrics["benchmark_total_return"] == pytest.approx(
        span.iloc[-1] / span.iloc[0] - 1, rel=1e-6
    )


def test_mixed_market_scan_aligns_benchmark_and_composite_by_date():
    bars = {
        "US.AAPL": make_daily_bars(
            n=700, start="2022-01-03", seed=1, price=150, tz="America/New_York"
        ),
        "HK.00700": make_daily_bars(n=700, start="2022-01-03", seed=2, price=300),
    }
    insts = {"US.AAPL": Instrument("US.AAPL"), "HK.00700": Instrument("HK.00700", lot_size=100)}
    hsi = Instrument("HK.800000", lot_size=1, security_type="IDX")
    result = run_on_bars(
        _cfg(symbols=list(bars)),
        create_strategy("macd"),
        bars,
        insts,
        benchmark=(make_daily_bars(n=700, start="2022-01-03", seed=3, price=20_000), hsi),
    )
    assert "beta" in result.books["US.AAPL"].metrics
    comp = result.books[COMPOSITE]
    assert not comp.equity.index.has_duplicates
    assert comp.metrics["periods_per_year"] < 300


def test_optimizer_reports_invalid_parameter_sets(tmp_path):
    from powerbacktest.engine.runner import LoadedData

    bars = {"HK.00700": make_daily_bars(n=600, start="2022-01-03", seed=1, price=300)}
    cfg = RunConfig.model_validate(
        {
            "backtest": {
                "strategy": "macd",
                "symbols": ["HK.00700"],
                "start": "2023-01-01",
                "end": "2024-03-31",
            },
            "optimize": {"grid": {"fast_period": [12, 30]}, "jobs": 1},
        }
    )
    result = optimize(
        cfg, LoadedData(bars, {"HK.00700": Instrument("HK.00700", lot_size=100)}, None, [])
    )
    assert len(result.table) == 1 and len(result.skipped) == 1
    assert result.summary()["skipped"] == 1


def test_btse_warmup_covers_point_in_time_window():
    assert create_strategy("btse").warmup_bars() == 250
    assert create_strategy("btse", {"zig_mode": "tdx"}).warmup_bars() == 25 + 30 + 1


class _PeekParams(StrategyParams):
    pass


class _Peek(Strategy):
    name = ""
    Params = _PeekParams
    seen: ClassVar[list[int]] = []

    def indicators(self, bars):
        return pd.DataFrame({"c": bars["close"]}, index=bars.index)

    def signals(self, bars, ind):
        return pd.Series(np.nan, index=bars.index)

    def on_bar(self, ctx: BarContext) -> float:
        assert len(ctx.indicators) == ctx.index + 1 and len(ctx.bars) == ctx.index + 1
        assert ctx.indicator("c") == ctx.close
        type(self).seen.append(ctx.index)
        return ctx.signal


def test_on_bar_sees_only_history():
    bars = bars_from_prices([10.0, 11, 12, 13])
    strat = _Peek()
    ind, sig = strat.run(bars)
    feed = SymbolFeed(
        HKI := Instrument("HK.00700", lot_size=100), bars, ind, sig.to_numpy(float), 0, 3
    )
    Simulator(
        [feed],
        strat,
        FREE,
        capital=1e4,
        execution=NO_SLIP,
        risk=RiskConfig(),
        portfolio=PortfolioConfig(),
        sizer="all_in",
        name=HKI.symbol,
    ).run()
    assert _Peek.seen == [0, 1, 2, 3]
