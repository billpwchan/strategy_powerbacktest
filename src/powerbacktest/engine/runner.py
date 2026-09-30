"""Run a configured backtest end to end.

``run_backtest`` loads data through the :class:`DataManager` and calls ``run_on_bars``; the
optimiser calls ``run_on_bars`` directly so data is loaded once per sweep.

Modes
-----
* ``scan``: every symbol is simulated on its own with the full initial capital. This answers
  "on which stocks does this strategy work". The ``COMPOSITE`` book averages the symbol books,
  i.e. it approximates splitting the capital equally with no rebalancing.
* ``portfolio``: all symbols share one cash balance and the sizer allocates it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from powerbacktest.analytics.metrics import align_series, book_metrics
from powerbacktest.config import CostsConfig, RunConfig
from powerbacktest.data.manager import DataManager
from powerbacktest.engine.simulator import (
    Simulator,
    SymbolFeed,
    bar_close_ns,
    buy_and_hold,
    display_index,
    is_intraday_index,
)
from powerbacktest.engine.types import Book
from powerbacktest.errors import DataError
from powerbacktest.market.costs import (
    CostModel,
    HKCostModel,
    MarketCostModel,
    SimpleCostModel,
    USCostModel,
)
from powerbacktest.market.instrument import Instrument
from powerbacktest.strategy.base import PlotSpec, Strategy
from powerbacktest.strategy.lookahead import LookaheadReport, check_lookahead
from powerbacktest.strategy.registry import create_strategy, load_strategy_paths
from powerbacktest.timeframe import Timeframe

log = logging.getLogger(__name__)

PORTFOLIO = "PORTFOLIO"
COMPOSITE = "COMPOSITE"


@dataclass
class SymbolView:
    instrument: Instrument
    bars: pd.DataFrame
    indicators: pd.DataFrame
    plots: tuple[PlotSpec, ...]


@dataclass
class BacktestResult:
    config: RunConfig
    strategy: dict[str, Any]
    mode: str
    books: dict[str, Book]
    primary: str
    symbols: dict[str, SymbolView]
    warnings: list[str] = field(default_factory=list)
    lookahead: LookaheadReport | None = None
    costs: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    @property
    def main(self) -> Book:
        return self.books[self.primary]

    def symbol_books(self) -> dict[str, Book]:
        return {k: v for k, v in self.books.items() if k not in (COMPOSITE, PORTFOLIO)}

    def summary(self) -> pd.DataFrame:
        cols = [
            "total_return",
            "cagr",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "trades",
            "win_rate",
            "profit_factor",
            "exposure",
            "total_fees",
            "buy_hold_return",
        ]
        rows = {name: {c: book.metrics.get(c) for c in cols} for name, book in self.books.items()}
        return pd.DataFrame.from_dict(rows, orient="index")


def build_cost_model(cfg: CostsConfig) -> CostModel:
    if cfg.model == "simple":
        return SimpleCostModel(cfg.simple.rate, cfg.simple.min_fee)
    return MarketCostModel(HKCostModel(cfg.hk), USCostModel(cfg.us))


def _window(bars: pd.DataFrame, tz: str, start: date, end: date) -> np.ndarray:
    dates = pd.DatetimeIndex(bars.index).tz_convert(tz).date
    return np.flatnonzero((dates >= start) & (dates <= end))


def make_feed(
    strategy: Strategy,
    bars: pd.DataFrame,
    instrument: Instrument,
    start: date,
    end: date,
    warnings: list[str],
) -> SymbolFeed | None:
    idx = _window(bars, instrument.tz, start, end)
    if len(idx) == 0:
        warnings.append(f"{instrument.symbol}: no bars between {start} and {end}; skipped")
        return None
    warmup = strategy.warmup_bars()
    first, last = max(int(idx[0]), warmup), int(idx[-1])
    if first > last:
        warnings.append(
            f"{instrument.symbol}: {len(bars)} bars are fewer than the {warmup}-bar warm-up; skipped"
        )
        return None
    if idx[0] < warmup:
        warnings.append(
            f"{instrument.symbol}: only {idx[0]} bars of history before {start}; trading starts "
            f"{bars.index[first].date()} after the {warmup}-bar warm-up"
        )
    trimmed = bars.iloc[: last + 1]
    ind, sig = strategy.run(trimmed)
    return SymbolFeed(instrument, trimmed, ind, sig.to_numpy(float), first, last)


def _by_date(series: pd.Series) -> pd.Series:
    """Re-key a daily series by its local trading date (as UTC midnight)."""
    local = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    out = pd.Series(
        series.to_numpy(), index=pd.DatetimeIndex(local, name="time").tz_localize("UTC")
    )
    return out[~out.index.duplicated(keep="last")]


def _composite(books: list[Book], name: str) -> Book:
    n = len(books)
    tzs = {str(pd.DatetimeIndex(b.equity.index).tz) for b in books}
    daily = not any(is_intraday_index(pd.DatetimeIndex(b.equity.index)) for b in books)
    # Mixed-market daily books are joined on trading date, not on instants, so an HK and a US
    # bar for the same date form one composite row.
    key = _by_date if len(tzs) > 1 and daily else (lambda s: s)
    index = key(books[0].equity).index
    for b in books[1:]:
        index = index.union(key(b.equity).index)

    def avg(attr: str, fill: float | None) -> pd.Series:
        frames = []
        for b in books:
            s = key(getattr(b, attr)).reindex(index).ffill()
            frames.append(s.fillna(b.initial_capital if fill is None else fill))
        return pd.concat(frames, axis=1).mean(axis=1)

    comp = Book(
        name=name,
        currency=books[0].currency,
        initial_capital=books[0].initial_capital,
        equity=avg("equity", None),
        cash=avg("cash", None),
        exposure=avg("exposure", 0.0),
        fills=[f for b in books for f in b.fills],
        trades=sorted((t for b in books for t in b.trades), key=lambda t: t.entry_time),
        symbols=[s for b in books for s in b.symbols],
    )
    if all(b.buy_hold is not None for b in books):
        comp.buy_hold = pd.concat(
            [
                key(b.buy_hold).reindex(index).ffill().fillna(b.initial_capital)
                for b in books
                if b.buy_hold is not None
            ],
            axis=1,
        ).mean(axis=1)
    comp.event_counts = {}
    for b in books:
        for k, v in b.event_counts.items():
            comp.event_counts[k] = comp.event_counts.get(k, 0) + v
    comp.events = [e for b in books for e in b.events][:500]
    comp.metrics["_scale"] = 1.0 / n
    return comp


def _index_benchmark(
    bars: pd.DataFrame, instrument: Instrument, start: date, end: date, capital: float
) -> pd.Series | None:
    idx = _window(bars, instrument.tz, start, end)
    if len(idx) == 0:
        return None
    sel = bars.iloc[idx]
    closes = pd.Series(sel["close"].to_numpy(float), index=bar_close_ns(sel.index, instrument))
    feed_like = SymbolFeed(
        instrument, sel, pd.DataFrame(index=sel.index), np.zeros(len(sel)), 0, len(sel) - 1
    )
    closes.index = display_index(closes.index.to_numpy(), [feed_like])
    return capital * closes / closes.iloc[0]


def _fallback_ppy(tf: Timeframe, feeds: list[SymbolFeed]) -> float:
    return tf.bars_per_year(feeds[0].instrument.market)


def run_on_bars(
    config: RunConfig,
    strategy: Strategy,
    bars_by_symbol: dict[str, pd.DataFrame],
    instruments: dict[str, Instrument],
    *,
    benchmark: tuple[pd.DataFrame, Instrument] | None = None,
    start: date | None = None,
    end: date | None = None,
    check_future: bool = True,
    extra_warnings: list[str] | None = None,
) -> BacktestResult:
    bt = config.backtest
    start = start or bt.start
    end = end or bt.end
    started = datetime.now(UTC).isoformat(timespec="seconds")
    warnings = list(extra_warnings or []) + strategy.warnings()
    feeds: list[SymbolFeed] = []
    for sym in bt.symbols:
        bars = bars_by_symbol.get(sym)
        if bars is None or bars.empty:
            warnings.append(f"{sym}: no data; skipped")
            continue
        feed = make_feed(strategy, bars, instruments[sym], start, end, warnings)
        if feed is not None:
            feeds.append(feed)
    if not feeds:
        raise DataError("No symbol has enough data to backtest: " + "; ".join(warnings))

    lookahead = None
    if check_future:
        probe = max(feeds, key=lambda f: f.last)
        lookahead = check_lookahead(strategy, probe.bars, checkpoints=6)
        if not lookahead.ok:
            warnings.append(f"{strategy.name}: {lookahead.summary()} on {probe.symbol}")

    costs = build_cost_model(config.costs)
    sizer = config.portfolio.resolved_sizer(bt.mode)
    capital = bt.initial_capital
    tf = bt.tf
    common = {
        "execution": config.execution,
        "risk": config.risk,
        "portfolio": config.portfolio,
        "sizer": sizer,
    }
    bench_series = None
    bench_name = None
    if benchmark is not None:
        bench_series = _index_benchmark(benchmark[0], benchmark[1], start, end, capital)
        bench_name = benchmark[1].name or benchmark[1].symbol
        if bench_series is None:
            warnings.append(f"Benchmark {benchmark[1].symbol}: no bars in range; using buy & hold")

    books: dict[str, Book] = {}
    if bt.mode == "portfolio":
        book = Simulator(feeds, strategy, costs, capital=capital, name=PORTFOLIO, **common).run()
        book.buy_hold = buy_and_hold(feeds, costs, capital, config.execution, book.equity.index)
        books[PORTFOLIO] = book
        primary = PORTFOLIO
    else:
        for feed in feeds:
            book = Simulator(
                [feed], strategy, costs, capital=capital, name=feed.symbol, **common
            ).run()
            book.buy_hold = buy_and_hold(
                [feed], costs, capital, config.execution, book.equity.index
            )
            books[feed.symbol] = book
        if len(feeds) > 1:
            books[COMPOSITE] = _composite(list(books.values()), COMPOSITE)
            primary = COMPOSITE
        else:
            primary = feeds[0].symbol

    fallback = _fallback_ppy(tf, feeds)
    for book in books.values():
        if bench_series is not None:
            # Rebased per book, so a symbol that starts trading late (IPO, short history) is
            # compared with the index over its own trading span only.
            book.benchmark = align_series(
                bench_series, pd.DatetimeIndex(book.equity.index), capital, rebase=True
            )
            book.benchmark_name = bench_name
        else:
            book.benchmark, book.benchmark_name = book.buy_hold, "Buy & hold"
        scale = book.metrics.pop("_scale", None)
        book.metrics = book_metrics(
            book, fallback_ppy=fallback, risk_free_rate=config.analytics.risk_free_rate
        )
        if scale is not None:
            for key in (
                "total_fees",
                "fees_pct_of_capital",
                "turnover",
                "realized_pnl",
                "unrealized_pnl",
                "expectancy",
                "avg_win",
                "avg_loss",
                "largest_win",
                "largest_loss",
            ):
                if isinstance(book.metrics.get(key), float):
                    book.metrics[key] *= scale
            book.metrics["composite_note"] = (
                "Average of the symbol books: equal capital per symbol, no rebalancing; money "
                "figures are scaled to one book's capital."
            )

    views = {
        f.symbol: SymbolView(
            instrument=f.instrument,
            bars=f.bars.iloc[f.first : f.last + 1],
            indicators=f.indicators.iloc[f.first : f.last + 1],
            plots=strategy.plots,
        )
        for f in feeds
    }
    return BacktestResult(
        config=config,
        strategy=strategy.describe(),
        mode=bt.mode,
        books=books,
        primary=primary,
        symbols=views,
        warnings=warnings,
        lookahead=lookahead,
        costs={"model": config.costs.model, **costs.describe()},
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


@dataclass
class LoadedData:
    bars: dict[str, pd.DataFrame]
    instruments: dict[str, Instrument]
    benchmark: tuple[pd.DataFrame, Instrument] | None
    warnings: list[str]


def load_data(config: RunConfig, data: DataManager, warmup_bars: int) -> LoadedData:
    bt = config.backtest
    tf = bt.tf
    instruments = data.instruments(bt.symbols, bt.lot_sizes)
    bars: dict[str, pd.DataFrame] = {}
    for sym in bt.symbols:
        try:
            bars[sym] = data.load_bars(sym, tf, bt.start, bt.end, warmup_bars=warmup_bars)
        except DataError as exc:
            data.warnings.append(f"{sym}: {exc}")
    benchmark = None
    if bt.benchmark:
        try:
            inst = data.instruments([bt.benchmark], {bt.benchmark: 1})[bt.benchmark]
            inst = Instrument(inst.symbol, lot_size=1, name=inst.name, security_type="IDX")
            benchmark = (data.load_bars(bt.benchmark, tf, bt.start, bt.end), inst)
        except DataError as exc:
            data.warnings.append(f"Benchmark {bt.benchmark}: {exc}")
    return LoadedData(bars, instruments, benchmark, list(data.warnings))


def run_backtest(
    config: RunConfig, data: DataManager, strategy: Strategy | None = None
) -> BacktestResult:
    if config.backtest.strategy_paths:
        load_strategy_paths(config.backtest.strategy_paths)
    strategy = strategy or create_strategy(config.backtest.strategy, config.backtest.params)
    loaded = load_data(config, data, strategy.warmup_bars())
    log.info(
        "Running %s on %d symbol(s), %s %s..%s, mode=%s",
        strategy.name,
        len(loaded.bars),
        config.backtest.timeframe,
        config.backtest.start,
        config.backtest.end,
        config.backtest.mode,
    )
    return run_on_bars(
        config,
        strategy,
        loaded.bars,
        loaded.instruments,
        benchmark=loaded.benchmark,
        check_future=config.backtest.check_lookahead,
        extra_warnings=loaded.warnings,
    )
