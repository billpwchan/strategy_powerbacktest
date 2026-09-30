"""Parameter search with out-of-sample validation.

Three modes, all on data loaded once:

* full period: rank every grid point on the whole backtest (fast, but in-sample only);
* ``split``: rank on the in-sample part, then report the top candidates' out-of-sample
  results next to their in-sample ones, so overfitting shows up as a gap;
* ``walk_forward_folds``: anchored walk-forward. Each fold picks the best parameters on its
  training window and is scored on the following, unseen slice; the stitched test slices are
  the honest estimate of the procedure's performance.

Indicators are always warmed up on the bars *before* each window, so a test slice starts
trading on day one without peeking at data inside it.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yaml

from powerbacktest.config import RunConfig
from powerbacktest.engine.runner import LoadedData, run_on_bars
from powerbacktest.errors import ConfigError, DataError, StrategyError
from powerbacktest.strategy.registry import create_strategy, get_strategy_class

log = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "trades",
    "win_rate",
    "profit_factor",
)


def parse_grid_spec(text: str) -> tuple[str, list[Any]]:
    """``fast_period=5:20:5`` (inclusive range), ``mode=cross,state`` or ``zig_pct=[5, 7.5]``."""
    if "=" not in text:
        raise ConfigError(f"Grid spec {text!r} must look like name=start:stop:step or name=a,b,c")
    name, raw = (part.strip() for part in text.split("=", 1))
    if raw.count(":") == 2 and "," not in raw:
        start, stop, step = (yaml.safe_load(x) for x in raw.split(":"))
        if not all(isinstance(x, (int, float)) for x in (start, stop, step)) or step <= 0:
            raise ConfigError(f"Bad range in {text!r}")
        n = math.floor((stop - start) / step + 1e-9) + 1
        values = [start + i * step for i in range(n)]
        if all(isinstance(x, int) for x in (start, stop, step)):
            values = [int(v) for v in values]
        else:
            values = [round(float(v), 10) for v in values]
        return name, values
    parsed = yaml.safe_load(raw if raw.startswith("[") else f"[{raw}]")
    if not isinstance(parsed, list) or not parsed:
        raise ConfigError(f"No values in grid spec {text!r}")
    return name, parsed


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    names = list(grid)
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(*(grid[n] for n in names))
    ]


def _score(metrics: dict[str, Any], metric: str) -> float:
    name = metric.lstrip("-")
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
        return -math.inf
    return -float(value) if metric.startswith("-") else float(value)


# ------------------------------------------------------------------ workers

_STATE: dict[str, Any] = {}


def _init_worker(config_json: str, loaded: LoadedData) -> None:
    _STATE["config"] = RunConfig.model_validate_json(config_json)
    _STATE["loaded"] = loaded


def _evaluate(task: tuple[dict[str, Any], date, date]) -> dict[str, Any]:
    params, start, end = task
    config: RunConfig = _STATE["config"]
    loaded: LoadedData = _STATE["loaded"]
    merged = {**config.backtest.params, **params}
    try:
        strategy = create_strategy(config.backtest.strategy, merged)
    except StrategyError as exc:
        return {"params": params, "error": str(exc).splitlines()[0]}
    try:
        result = run_on_bars(
            config,
            strategy,
            loaded.bars,
            loaded.instruments,
            benchmark=loaded.benchmark,
            start=start,
            end=end,
            check_future=False,
        )
    except DataError as exc:
        return {"params": params, "error": str(exc)}
    main = result.main
    return {"params": params, "metrics": dict(main.metrics), "equity": main.equity}


def _run_tasks(
    config: RunConfig, loaded: LoadedData, tasks: list[tuple[dict[str, Any], date, date]], jobs: int
) -> list[dict[str, Any]]:
    workers = jobs or os.cpu_count() or 1
    if workers <= 1 or len(tasks) <= 2:
        _init_worker(config.model_dump_json(), loaded)
        return [_evaluate(t) for t in tasks]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        initializer=_init_worker,
        initargs=(config.model_dump_json(), loaded),
    ) as pool:
        return list(pool.map(_evaluate, tasks, chunksize=max(1, len(tasks) // (workers * 4))))


# ------------------------------------------------------------------ results


@dataclass
class OptimizeResult:
    metric: str
    mode: str
    table: pd.DataFrame
    folds: list[dict[str, Any]] = field(default_factory=list)
    oos_equity: pd.Series | None = None
    oos_return: float | None = None
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "metric": self.metric,
            "mode": self.mode,
            "candidates": len(self.table),
        }
        if self.folds:
            out["folds"] = self.folds
        if self.oos_return is not None:
            out["walk_forward_oos_return"] = self.oos_return
        if self.skipped:
            out["skipped"] = len(self.skipped)
        return out


def _rows(
    results: Iterable[dict[str, Any]], prefix: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, skipped = [], []
    for r in results:
        if "error" in r:
            skipped.append(r)
            continue
        row = {f"param_{k}": v for k, v in r["params"].items()}
        row.update({f"{prefix}{c}": r["metrics"].get(c) for c in METRIC_COLUMNS})
        rows.append(row)
    return rows, skipped


def _filter(results: list[dict[str, Any]], min_trades: int) -> list[dict[str, Any]]:
    if not min_trades:
        return results
    return [r for r in results if "error" in r or (r["metrics"].get("trades") or 0) >= min_trades]


def _best(results: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return None
    return max(ok, key=lambda r: _score(r["metrics"], metric))


def optimize(config: RunConfig, loaded: LoadedData) -> OptimizeResult:
    opt = config.optimize
    get_strategy_class(config.backtest.strategy)  # fail fast on unknown strategy
    combos = expand_grid(opt.grid)
    bt = config.backtest
    metric = opt.metric
    log.info("Optimising %s over %d parameter sets (metric %s)", bt.strategy, len(combos), metric)

    if opt.walk_forward_folds:
        return _walk_forward(config, loaded, combos)

    if opt.split:
        if not bt.start < opt.split <= bt.end:
            raise ConfigError(f"optimize.split {opt.split} must fall inside {bt.start}..{bt.end}")
        is_end = opt.split - timedelta(days=1)
        in_sample = _filter(
            _run_tasks(config, loaded, [(c, bt.start, is_end) for c in combos], opt.jobs),
            opt.min_trades,
        )
        ranked = sorted(
            (r for r in in_sample if "error" not in r),
            key=lambda r: _score(r["metrics"], metric),
            reverse=True,
        )
        top = ranked[: opt.top]
        oos = _run_tasks(config, loaded, [(r["params"], opt.split, bt.end) for r in top], opt.jobs)
        is_rows, skipped = _rows(ranked, "is_")
        oos_rows, _ = _rows(oos, "oos_")
        table = pd.DataFrame(is_rows)
        if oos_rows:
            oos_frame = pd.DataFrame(oos_rows)
            keys = [c for c in oos_frame.columns if c.startswith("param_")]
            table = table.merge(oos_frame, on=keys, how="left")
        return OptimizeResult(metric, "split", table, skipped=skipped)

    results = _filter(
        _run_tasks(config, loaded, [(c, bt.start, bt.end) for c in combos], opt.jobs),
        opt.min_trades,
    )
    ranked = sorted(
        (r for r in results if "error" not in r),
        key=lambda r: _score(r["metrics"], metric),
        reverse=True,
    )
    rows, skipped = _rows(ranked, "")
    return OptimizeResult(metric, "full", pd.DataFrame(rows), skipped=skipped)


def fold_bounds(start: date, end: date, folds: int) -> list[tuple[date, date, date, date]]:
    """(train_start, train_end, test_start, test_end) for an anchored walk-forward."""
    edges = pd.date_range(
        pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), periods=folds + 2
    )
    days = [e.date() for e in edges]
    out = []
    for i in range(1, folds + 1):
        test_start = days[i]
        test_end = days[i + 1] - timedelta(days=1)
        out.append((start, test_start - timedelta(days=1), test_start, min(test_end, end)))
    return out


def _walk_forward(
    config: RunConfig, loaded: LoadedData, combos: list[dict[str, Any]]
) -> OptimizeResult:
    opt = config.optimize
    bt = config.backtest
    assert opt.walk_forward_folds is not None
    folds = []
    rows: list[dict[str, Any]] = []
    stitched: list[pd.Series] = []
    skipped: list[dict[str, Any]] = []
    for n, (tr_s, tr_e, te_s, te_e) in enumerate(
        fold_bounds(bt.start, bt.end, opt.walk_forward_folds), 1
    ):
        train = _filter(
            _run_tasks(config, loaded, [(c, tr_s, tr_e) for c in combos], opt.jobs), opt.min_trades
        )
        best = _best(train, opt.metric)
        if best is None:
            folds.append(
                {
                    "fold": n,
                    "train": [tr_s.isoformat(), tr_e.isoformat()],
                    "error": "no valid parameter set",
                }
            )
            continue
        test = _run_tasks(config, loaded, [(best["params"], te_s, te_e)], 1)[0]
        info: dict[str, Any] = {
            "fold": n,
            "train": [tr_s.isoformat(), tr_e.isoformat()],
            "test": [te_s.isoformat(), te_e.isoformat()],
            "params": best["params"],
            "train_score": _score(best["metrics"], opt.metric),
        }
        if "error" in test:
            info["error"] = test["error"]
        else:
            info["test_metrics"] = {c: test["metrics"].get(c) for c in METRIC_COLUMNS}
            stitched.append(test["equity"] / bt.initial_capital)
        folds.append(info)
        row = {"fold": n, **{f"param_{k}": v for k, v in best["params"].items()}}
        row.update({f"train_{c}": best["metrics"].get(c) for c in METRIC_COLUMNS})
        if "test_metrics" in info:
            row.update({f"test_{c}": v for c, v in info["test_metrics"].items()})
        rows.append(row)
        skipped.extend(r for r in train if "error" in r)
    oos = None
    oos_return = None
    if stitched:
        level = 1.0
        parts = []
        for s in stitched:
            parts.append(s * level)
            level = float(parts[-1].iloc[-1])
        oos = pd.concat(parts)
        oos_return = float(oos.iloc[-1] - 1)
        oos = oos[~oos.index.duplicated(keep="last")] * bt.initial_capital
    return OptimizeResult(
        opt.metric,
        "walk_forward",
        pd.DataFrame(rows),
        folds=folds,
        oos_equity=oos,
        oos_return=oos_return,
        skipped=skipped,
    )


def summarize_table(table: pd.DataFrame, top: int) -> str:
    if table.empty:
        return "(no results)"
    view = table.head(top).copy()
    for col in view.columns:
        if view[col].dtype.kind == "f":
            view[col] = view[col].map(
                lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4g}"
            )
    return view.to_string(index=False)
