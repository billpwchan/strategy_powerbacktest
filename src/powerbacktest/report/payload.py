"""Turn a :class:`BacktestResult` into the JSON document the HTML report renders.

The same document is written as ``result.json`` so a report can be re-rendered, diffed, or
loaded into a notebook without re-running the backtest. Times are encoded as the exchange's
local wall-clock time in epoch seconds, which is what the chart library displays.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from powerbacktest import __version__
from powerbacktest.analytics.metrics import bar_returns, drawdown, drawdown_periods, monthly_returns
from powerbacktest.config import dump_config
from powerbacktest.engine.runner import COMPOSITE, PORTFOLIO, BacktestResult
from powerbacktest.engine.types import Book

SCHEMA_VERSION = 1


def clean(value: Any) -> Any:
    """JSON-safe: NaN -> None, inf -> "inf", numpy scalars -> Python, timestamps -> ISO."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def wall_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Local wall-clock time as if it were UTC, in whole seconds."""
    naive = index.tz_localize(None) if index.tz is not None else index
    return naive.as_unit("s").to_numpy().astype("int64")


def _series_points(series: pd.Series, max_points: int | None = None) -> list[list[float]]:
    s = series.dropna()
    if max_points and len(s) > max_points:
        step = math.ceil(len(s) / max_points)
        keep = np.r_[np.arange(0, len(s), step), len(s) - 1]
        s = s.iloc[np.unique(keep)]
    secs = wall_seconds(pd.DatetimeIndex(s.index))
    return [[int(t), round(float(v), 6)] for t, v in zip(secs, s.to_numpy(float), strict=True)]


def _book_payload(book: Book, max_points: int) -> dict[str, Any]:
    monthly = monthly_returns(book.equity, book.initial_capital)
    bench = None
    if book.benchmark is not None:
        bench = book.benchmark.reindex(book.equity.index).ffill().fillna(book.initial_capital)
    return {
        "name": book.name,
        "currency": book.currency,
        "initial_capital": book.initial_capital,
        "symbols": book.symbols,
        "metrics": clean(book.metrics),
        "equity": _series_points(book.equity, max_points),
        "benchmark": _series_points(bench, max_points) if bench is not None else None,
        "benchmark_name": book.benchmark_name,
        "drawdown": _series_points(drawdown(book.equity, book.initial_capital), max_points),
        "exposure": _series_points(book.exposure, max_points),
        "monthly": {
            "years": [int(y) for y in monthly.index] if len(monthly) else [],
            "rows": clean(monthly.to_numpy().tolist()) if len(monthly) else [],
        },
        "drawdown_periods": clean(
            [
                {
                    **p,
                    "start": p["start"].isoformat(),
                    "trough": p["trough"].isoformat(),
                    "end": p["end"].isoformat() if p["end"] is not None else None,
                }
                for p in drawdown_periods(book.equity, book.initial_capital, top=5)
            ]
        ),
        "trades": clean([t.to_dict() for t in book.trades]),
        "events": {"counts": book.event_counts, "sample": book.events[:100]},
    }


def _correlation(result: BacktestResult) -> dict[str, Any] | None:
    books = result.symbol_books()
    if result.mode != "scan" or len(books) < 2:
        return None
    rets = pd.concat(
        {name: bar_returns(b.equity, b.initial_capital) for name, b in books.items()}, axis=1
    )
    corr = rets.corr(min_periods=20)
    return {"symbols": list(corr.columns), "matrix": clean(corr.to_numpy().tolist())}


def _symbol_payload(result: BacktestResult, max_bars: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fills_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for name, book in result.books.items():
        if name == COMPOSITE:
            continue
        for f in book.fills:
            fills_by_symbol.setdefault(f.symbol, []).append(
                {
                    "time": int(wall_seconds(pd.DatetimeIndex([f.time]))[0]),
                    "side": f.side,
                    "price": f.price,
                    "quantity": f.quantity,
                    "reason": f.reason,
                    "fees": f.fees.total,
                }
            )
    for sym, view in result.symbols.items():
        bars = view.bars.iloc[-max_bars:]
        ind = view.indicators.loc[bars.index]
        secs = wall_seconds(pd.DatetimeIndex(bars.index))
        ohlc = np.column_stack(
            [secs, bars[["open", "high", "low", "close", "volume"]].to_numpy(float)]
        )
        plots = []
        for spec in view.plots:
            if spec.column not in ind.columns:
                continue
            values = ind[spec.column].to_numpy(float)
            plots.append(
                {
                    "column": spec.column,
                    "label": spec.label or spec.column,
                    "pane": spec.pane,
                    "kind": spec.kind,
                    "data": [
                        [int(t), round(float(v), 6)]
                        for t, v in zip(secs, values, strict=True)
                        if not math.isnan(v)
                    ],
                }
            )
        inst = view.instrument
        out[sym] = {
            "symbol": sym,
            "name": inst.name,
            "lot_size": inst.lot_size,
            "currency": inst.currency,
            "security_type": inst.security_type,
            "truncated": len(view.bars) > len(bars),
            "ohlc": [[int(r[0]), *(round(float(x), 6) for x in r[1:])] for r in ohlc],
            "plots": plots,
            "fills": fills_by_symbol.get(sym, []),
        }
    return out


def build_payload(result: BacktestResult) -> dict[str, Any]:
    cfg = result.config
    bt = cfg.backtest
    max_bars = cfg.report.max_bars_per_symbol
    books = {name: _book_payload(book, max_points=max_bars) for name, book in result.books.items()}
    order = [result.primary] + [n for n in books if n != result.primary]
    intraday = bt.tf.is_intraday
    return {
        "schema": SCHEMA_VERSION,
        "meta": {
            "title": cfg.report.title
            or f"{result.strategy.get('title') or result.strategy['name']}",
            "strategy": result.strategy,
            "mode": result.mode,
            "timeframe": bt.timeframe,
            "intraday": intraday,
            "start": bt.start.isoformat(),
            "end": bt.end.isoformat(),
            "symbols": bt.symbols,
            "initial_capital": bt.initial_capital,
            "adjust": cfg.data.adjust,
            "execution": cfg.execution.model_dump(mode="json"),
            "risk": cfg.risk.model_dump(mode="json"),
            "portfolio": {
                **cfg.portfolio.model_dump(mode="json"),
                "sizer": cfg.portfolio.resolved_sizer(bt.mode),
            },
            "costs": clean(result.costs),
            "risk_free_rate": cfg.analytics.risk_free_rate,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "version": __version__,
            "config_yaml": dump_config(cfg),
        },
        "warnings": result.warnings,
        "lookahead": (
            {"ok": result.lookahead.ok, "summary": result.lookahead.summary()}
            if result.lookahead is not None
            else None
        ),
        "primary": result.primary,
        "book_order": order,
        "portfolio_key": PORTFOLIO,
        "composite_key": COMPOSITE,
        "books": books,
        "symbols": _symbol_payload(result, max_bars),
        "correlation": _correlation(result),
    }
