from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from helpers import bars_from_prices, make_daily_bars
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from powerbacktest.config import ExecutionConfig, PortfolioConfig, RiskConfig, RunConfig
from powerbacktest.engine.runner import COMPOSITE, PORTFOLIO, run_on_bars
from powerbacktest.engine.simulator import Simulator, SymbolFeed, bar_close_ns
from powerbacktest.market.costs import HKCostModel, SimpleCostModel, USCostModel
from powerbacktest.market.instrument import Instrument
from powerbacktest.market.ticks import hk_tick_size
from powerbacktest.strategy.base import Strategy, StrategyParams, events


class FixedParams(StrategyParams):
    signals: tuple[float | None, ...] = ()


class Fixed(Strategy):
    """Emits a predetermined signal list (None = no opinion)."""

    name = ""
    Params = FixedParams

    def indicators(self, bars):
        return pd.DataFrame(index=bars.index)

    def signals(self, bars, ind):
        values = [np.nan if v is None else v for v in self.params.signals]
        values += [np.nan] * (len(bars) - len(values))
        return pd.Series(values[: len(bars)], index=bars.index, dtype=float)


HK700 = Instrument("HK.00700", lot_size=100)
FREE = SimpleCostModel(0.0)
NO_SLIP = ExecutionConfig(slippage_ticks=0)


def simulate(
    bars,
    signals,
    *,
    inst=HK700,
    costs=FREE,
    capital=10_000.0,
    ex=NO_SLIP,
    risk=None,
    sizer="all_in",
    pf=None,
):
    strat = Fixed(signals=signals)
    ind, sig = strat.run(bars)
    feed = SymbolFeed(inst, bars, ind, sig.to_numpy(float), 0, len(bars) - 1)
    sim = Simulator(
        [feed],
        strat,
        costs,
        capital=capital,
        execution=ex,
        risk=risk or RiskConfig(),
        portfolio=pf or PortfolioConfig(),
        sizer=sizer,
        name="t",
    )
    return sim.run()


# ---------------------------------------------------------------- costs


def test_hk_fees_by_hand_2024():
    fees = HKCostModel().fees("BUY", 800, 12.0, HK700, date(2024, 1, 4))
    assert fees.commission == 3.00  # 0.03% = 2.88 < HK$3 minimum
    assert fees.platform == 15.0
    assert fees.stamp_duty == 10.0  # 0.1% of 9,600 = 9.6, rounded up
    assert fees.exchange == pytest.approx(0.54)
    assert fees.levies == pytest.approx(0.26 + 0.01)
    assert fees.settlement == 2.00  # 0.002% = 0.19 < HK$2 minimum
    assert fees.total == pytest.approx(30.81)


def test_hk_fee_schedule_changes_by_date():
    m = HKCostModel()
    assert m.fees("SELL", 800, 12.0, HK700, date(2022, 6, 1)).stamp_duty == 13.0  # 0.13%
    old = m.fees("SELL", 800, 12.0, HK700, date(2022, 6, 1))
    assert old.exchange == pytest.approx(0.48 + 0.5)  # 0.005% + HK$0.50 tariff
    assert m.fees("SELL", 800, 12.0, HK700, date(2025, 7, 2)).settlement == pytest.approx(0.40)
    etf = Instrument("HK.02800", lot_size=500, security_type="ETF")
    assert m.fees("BUY", 500, 25.0, etf, date(2024, 1, 4)).stamp_duty == 0.0


def test_us_fees_minimums_and_sell_side_levies():
    aapl = Instrument("US.AAPL")
    m = USCostModel()
    buy = m.fees("BUY", 10, 200.0, aapl, date(2024, 6, 3))
    assert buy.commission == 0.99 and buy.platform == 1.0 and buy.levies == 0.0
    sell = m.fees("SELL", 1000, 200.0, aapl, date(2024, 6, 3))
    assert sell.commission == pytest.approx(4.9) and sell.platform == pytest.approx(5.0)
    assert sell.levies == pytest.approx(round(200_000 * 27.8e-6, 2) + round(1000 * 0.000166, 2))


def test_hk_tick_table_by_date():
    assert hk_tick_size(15.0, date(2024, 1, 2)) == 0.02
    assert hk_tick_size(15.0, date(2025, 8, 4)) == 0.01
    assert hk_tick_size(35.0, date(2025, 9, 1)) == 0.02
    assert hk_tick_size(5.0, date(2026, 8, 3)) == 0.005
    assert hk_tick_size(350.0, date(2024, 1, 2)) == 0.2


# ---------------------------------------------------------------- fills


def test_signal_fills_at_next_open_whole_lots():
    bars = bars_from_prices([10, 11, 12, 13, 14, 15], [10.5, 11.5, 12.5, 13.5, 14.5, 15.5])
    book = simulate(bars, [None, 1, None, 0])
    buy, sell = book.fills
    assert (buy.side, buy.bar_index, buy.price, buy.quantity) == ("BUY", 2, 12.0, 800)
    assert (sell.side, sell.bar_index, sell.price, sell.quantity) == ("SELL", 4, 14.0, 800)
    assert book.cash.iloc[-1] == pytest.approx(400 + 800 * 14)
    assert book.equity.iloc[2] == pytest.approx(400 + 800 * 12.5)
    trade = book.trades[0]
    assert (
        trade.pnl == pytest.approx(800 * 2)
        and trade.bars_held == 2
        and trade.exit_reason == "signal"
    )


def test_fees_shrink_quantity_by_a_lot():
    bars = bars_from_prices([12.0] * 5)
    book = simulate(bars, [None, 1], costs=HKCostModel(), capital=9_620.0)
    assert book.fills[0].quantity == 700
    assert book.cash.iloc[-1] >= 0


def test_slippage_in_ticks_and_close_mode():
    bars = bars_from_prices([12.0, 12.0, 12.0, 12.0], [12.2] * 4, highs=[13] * 4, lows=[11] * 4)
    book = simulate(
        bars, [1, 0], ex=ExecutionConfig(slippage_ticks=2, entry_requires_fresh_signal=False)
    )
    assert book.fills[0].price == pytest.approx(12.0 + 2 * 0.02)
    assert book.fills[1].price == pytest.approx(12.0 - 2 * 0.02)
    close_mode = simulate(bars, [None, 1, 0], ex=ExecutionConfig(slippage_ticks=0, fill="close"))
    assert [(f.bar_index, f.price) for f in close_mode.fills] == [(1, 12.2), (2, 12.2)]


def test_fresh_signal_required_for_state_signals():
    bars = bars_from_prices([10.0] * 6)
    assert simulate(bars, [1, 1, 1, 1, 1, 1]).fills == []
    fills = simulate(bars, [1, 1, 0, 1, 1, 1]).fills
    assert [f.bar_index for f in fills] == [
        4
    ]  # unblocked by the 0 at bar 2, signal at 3, fill at 4
    relaxed = simulate(
        bars, [1] * 6, ex=ExecutionConfig(slippage_ticks=0, entry_requires_fresh_signal=False)
    )
    assert relaxed.fills[0].bar_index == 1


def test_stop_loss_intrabar_and_gap():
    bars = bars_from_prices(
        [100, 100, 99, 98], [100, 100, 97, 98], highs=[101, 101, 100, 99], lows=[99, 99, 94, 97]
    )
    book = simulate(
        bars,
        [1],
        ex=ExecutionConfig(slippage_ticks=0, entry_requires_fresh_signal=False),
        risk=RiskConfig(stop_loss=0.05),
        capital=100_000,
    )
    sell = book.fills[1]
    assert (sell.bar_index, sell.price, sell.reason) == (2, 95.0, "stop_loss")
    gap = bars_from_prices([100, 100, 90, 91], highs=[101, 101, 92, 92], lows=[99, 99, 89, 90])
    book = simulate(
        gap,
        [1],
        ex=ExecutionConfig(slippage_ticks=0, entry_requires_fresh_signal=False),
        risk=RiskConfig(stop_loss=0.05),
        capital=100_000,
    )
    assert book.fills[1].price == 90.0


def test_take_profit_trailing_and_max_hold():
    ex = ExecutionConfig(slippage_ticks=0, entry_requires_fresh_signal=False)
    bars = bars_from_prices(
        [100, 100, 105, 108], highs=[100, 101, 112, 109], lows=[99, 99, 104, 107]
    )
    tp = simulate(bars, [1], ex=ex, risk=RiskConfig(take_profit=0.1), capital=100_000)
    assert tp.fills[1].price == pytest.approx(110.0) and tp.fills[1].reason == "take_profit"

    trail_bars = bars_from_prices(
        [100, 100, 110, 115, 104], highs=[100, 101, 120, 116, 106], lows=[99, 99, 109, 113, 100]
    )
    tr = simulate(trail_bars, [1], ex=ex, risk=RiskConfig(trailing_stop=0.1), capital=100_000)
    # peak 120 -> stop 108; bar 4 gaps open at 104, below the stop, so it fills at the open
    assert tr.fills[1].reason == "trailing_stop" and tr.fills[1].price == pytest.approx(104.0)

    flat = bars_from_prices([10.0] * 8)
    mh = simulate(flat, [1], ex=ex, risk=RiskConfig(max_holding_bars=3))
    assert [f.bar_index for f in mh.fills] == [1, 5]
    assert mh.fills[1].reason == "max_hold"


def test_orders_after_last_bar_expire_and_open_trade_is_marked():
    bars = bars_from_prices([10.0, 11.0, 12.0], [10.0, 11.0, 12.5])
    book = simulate(bars, [None, 1, None])
    assert book.trades[0].is_open and book.trades[0].mark_price == 12.5
    expired = simulate(bars, [None, None, 1])
    assert (
        expired.fills == [] and expired.event_counts == {"order_expired": 1}
    ) or expired.event_counts == {}


def test_liquidate_at_end_realises_everything():
    bars = bars_from_prices([10.0, 11.0, 12.0], [10.0, 11.0, 12.5])
    book = simulate(bars, [None, 1], ex=ExecutionConfig(slippage_ticks=0, liquidate_at_end=True))
    assert not book.trades[0].is_open and book.trades[0].exit_reason == "end"
    assert book.equity.iloc[-1] == pytest.approx(book.cash.iloc[-1])


def test_portfolio_equal_weight_respects_max_positions():
    insts = [Instrument(f"HK.0000{i}", lot_size=100) for i in (1, 2, 3)]
    bars = bars_from_prices([10.0] * 6)
    strat = Fixed(signals=[None, 1])
    feeds = []
    for inst in insts:
        ind, sig = strat.run(bars)
        feeds.append(SymbolFeed(inst, bars, ind, sig.to_numpy(float), 0, len(bars) - 1))
    sim = Simulator(
        feeds,
        strat,
        FREE,
        capital=30_000.0,
        execution=NO_SLIP,
        risk=RiskConfig(),
        portfolio=PortfolioConfig(max_positions=2),
        sizer="equal_weight",
        name="p",
    )
    book = sim.run()
    assert [f.symbol for f in book.fills] == ["HK.00001", "HK.00002"]
    assert all(f.quantity == 1500 for f in book.fills)
    assert book.event_counts.get("buy_skipped") == 1


def test_daily_close_time_orders_hk_before_us():
    hk = bars_from_prices([1.0, 1.0], tz="Asia/Hong_Kong")
    us = bars_from_prices([1.0, 1.0], tz="America/New_York")
    hk_ns = bar_close_ns(hk.index, HK700)
    us_ns = bar_close_ns(us.index, Instrument("US.AAPL"))
    assert hk_ns[0] < us_ns[0] < hk_ns[1]


# ------------------------------------------------------------ invariants


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    lot=st.sampled_from([1, 100, 500, 2000]),
    stop=st.one_of(st.none(), st.floats(0.02, 0.2)),
    fill=st.sampled_from(["next_open", "close"]),
)
def test_accounting_invariants(seed, lot, stop, fill):
    rng = np.random.default_rng(seed)
    bars = make_daily_bars(n=120, seed=seed, price=float(rng.uniform(1, 400)))
    sig = rng.choice([np.nan, 0.0, 1.0], size=len(bars), p=[0.7, 0.15, 0.15]).tolist()
    sig = [None if np.isnan(s) else s for s in sig]
    inst = Instrument("HK.00005", lot_size=lot)
    ex = ExecutionConfig(fill=fill, slippage_ticks=1, liquidate_at_end=True)
    book = simulate(
        bars,
        sig,
        inst=inst,
        costs=HKCostModel(),
        capital=200_000.0,
        ex=ex,
        risk=RiskConfig(stop_loss=stop),
    )
    assert (book.cash >= -1e-6).all()
    assert all(f.quantity % lot == 0 and f.quantity > 0 for f in book.fills)
    for f in book.fills:
        assert (
            bars["low"].iat[f.bar_index] - 1e-9 <= f.price <= bars["high"].iat[f.bar_index] + 1e-9
        )
    realized = sum(t.pnl for t in book.trades)
    assert realized == pytest.approx(book.equity.iloc[-1] - 200_000.0, abs=1e-6)
    if fill == "next_open":
        decided = {i for i, s in enumerate(sig) if s is not None}
        for f in book.fills:
            if f.reason == "signal":
                assert f.bar_index - 1 in decided


# ------------------------------------------------------------------ runner


def _config(mode="scan", **bt):
    base = {
        "backtest": {
            "strategy": "macd",
            "symbols": ["HK.00700", "HK.09988", "HK.00005"],
            "start": "2023-01-01",
            "end": "2024-06-30",
            "mode": mode,
            **bt,
        }
    }
    return RunConfig.model_validate(base)


def _universe():
    bars = {
        "HK.00700": make_daily_bars(n=700, start="2022-01-03", seed=1, price=300),
        "HK.09988": make_daily_bars(n=700, start="2022-01-03", seed=2, price=90),
        "HK.00005": make_daily_bars(n=700, start="2022-01-03", seed=3, price=40),
    }
    insts = {s: Instrument(s, lot_size=lot) for s, lot in zip(bars, (100, 100, 400), strict=True)}
    return bars, insts


@pytest.mark.parametrize("strategy", ["macd", "ma_cross", "kdj", "rsi", "btse", "leg"])
def test_run_on_bars_scan_mode_all_strategies(strategy):
    from powerbacktest.strategy import create_strategy

    bars, insts = _universe()
    cfg = _config(strategy=strategy)
    result = run_on_bars(cfg, create_strategy(strategy), bars, insts)
    assert result.primary == COMPOSITE
    assert set(result.books) == {"HK.00700", "HK.09988", "HK.00005", COMPOSITE}
    for name, book in result.books.items():
        assert book.equity.index[0].date() >= date(2023, 1, 1), name
        assert book.equity.index[-1].date() <= date(2024, 6, 30)
        assert "sharpe" in book.metrics and "buy_hold_return" in book.metrics
    assert result.lookahead is not None and result.lookahead.ok


def test_run_on_bars_portfolio_mode_and_index_benchmark():
    from powerbacktest.strategy import create_strategy

    bars, insts = _universe()
    cfg = _config(
        mode="portfolio", strategy="ma_cross", params={"short_window": 5, "long_window": 20}
    )
    hsi = Instrument("HK.800000", lot_size=1, security_type="IDX", name="HSI")
    result = run_on_bars(
        cfg,
        create_strategy("ma_cross", cfg.backtest.params),
        bars,
        insts,
        benchmark=(make_daily_bars(n=700, start="2022-01-03", seed=9, price=20000), hsi),
    )
    book = result.books[PORTFOLIO]
    assert result.primary == PORTFOLIO and book.benchmark_name == "HSI"
    assert "beta" in book.metrics and book.metrics["trades"] > 0
    assert (book.cash >= -1e-6).all()


def test_warmup_warning_when_history_short():
    from powerbacktest.strategy import create_strategy

    bars, insts = _universe()
    short = {s: b.loc[b.index >= "2022-12-15"] for s, b in bars.items()}
    result = run_on_bars(_config(), create_strategy("macd"), short, insts)
    assert any("warm-up" in w for w in result.warnings)
    first_fill = min(f.time for b in result.symbol_books().values() for f in b.fills)
    assert first_fill.date() > date(2023, 4, 1)
    _ = events
