"""Command line interface: ``pbt``.

pbt run -c configs/example.yaml                  run a backtest, write the report
pbt run -c cfg.yaml --symbols HK.00700 -p fast_period=10 --offline
pbt optimize -c cfg.yaml --grid fast_period=5:20:5 --grid slow_period=20:60:10 --split 2024-01-01
pbt data fetch -c cfg.yaml                        warm the local cache from OpenD
pbt data import export.csv --symbol HK.00700 --timeframe DAY --adjust qfq
pbt data import-ticks /data/sqlite/HK/2026*.db --symbol HK.00700
pbt data list | pbt quota | pbt strategies | pbt report reports/<run> | pbt init
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import webbrowser
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from powerbacktest import __version__
from powerbacktest.config import RunConfig, load_config, parse_override
from powerbacktest.data.importers import read_bar_files, ticks_to_minute_bars
from powerbacktest.data.manager import DataManager
from powerbacktest.data.source import FutuSource
from powerbacktest.data.store import Coverage, ParquetStore, SeriesKey
from powerbacktest.errors import ConfigError, PowerBacktestError
from powerbacktest.logging_setup import setup_logging
from powerbacktest.market.instrument import MARKETS, normalize_symbol, parse_symbol
from powerbacktest.timeframe import Timeframe

log = logging.getLogger("powerbacktest.cli")


# --------------------------------------------------------------- arguments


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="YAML config file (see configs/example.yaml)")
    p.add_argument("--strategy", help="Strategy name, module:Class or path.py:Class")
    p.add_argument("--params", help="Strategy parameters as JSON, merged over the config")
    p.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="One strategy parameter (repeatable)",
    )
    p.add_argument("--symbols", nargs="+", help="Futu codes, e.g. HK.00700 US.AAPL")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument(
        "--timeframe", help="1M 5M 15M 30M 60M 2H 4H DAY WEEK MON (any minute multiple works)"
    )
    p.add_argument("--mode", choices=["scan", "portfolio"])
    p.add_argument("--capital", type=float, help="Initial capital")
    p.add_argument("--benchmark", help="Benchmark symbol, e.g. HK.800000 (Hang Seng Index)")
    p.add_argument("--adjust", choices=["qfq", "hfq", "none"], help="Price adjustment")
    p.add_argument("--data-dir", help="Local cache directory")
    p.add_argument(
        "--offline", action="store_true", help="Use cached data only; never contact OpenD"
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="DOTTED=VALUE",
        help="Any config field, e.g. risk.stop_loss=0.08",
    )


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping = {
        "strategy": "backtest.strategy",
        "symbols": "backtest.symbols",
        "start": "backtest.start",
        "end": "backtest.end",
        "timeframe": "backtest.timeframe",
        "mode": "backtest.mode",
        "capital": "backtest.initial_capital",
        "benchmark": "backtest.benchmark",
        "adjust": "data.adjust",
        "data_dir": "data.dir",
    }
    for attr, dotted in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            out[dotted] = value
    if getattr(args, "offline", False):
        out["data.offline"] = True
    params_json = getattr(args, "params", None)
    if params_json:
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"--params is not valid JSON: {exc}") from exc
        for k, v in params.items():
            out[f"backtest.params.{k}"] = v
    for item in getattr(args, "param", []) or []:
        k, v = parse_override(item)
        out[f"backtest.params.{k}"] = v
    for item in getattr(args, "set", []) or []:
        k, v = parse_override(item)
        out[k] = v
    return out


def _load(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> RunConfig:
    overrides = _overrides(args)
    overrides.update(extra or {})
    config = load_config(args.config, overrides)
    setup_logging(config.logging.level, config.logging.file)
    return config


def _data_manager(config: RunConfig) -> DataManager:
    f = config.futu

    def factory() -> FutuSource:
        return FutuSource(
            f.host,
            f.port,
            rate_limit=f.rate_limit,
            rate_window=f.rate_window_seconds,
            max_retries=f.max_retries,
        )

    return DataManager(
        ParquetStore(config.data.dir),
        None if config.data.offline else factory,
        adjust=config.data.adjust,
        offline=config.data.offline,
        instrument_ttl_days=config.data.instrument_ttl_days,
    )


# ---------------------------------------------------------------- commands


def _fmt(v: Any, pct: bool = False, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float) and v in (float("inf"), float("-inf")):
        return "∞"
    if pct:
        return f"{v * 100:+.2f}%" if signed else f"{v * 100:.2f}%"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def print_summary(result: Any) -> None:
    signed = {"total_return", "cagr", "buy_hold_return"}
    pct_cols = signed | {"max_drawdown", "win_rate", "exposure"}
    frame = result.summary()
    shown = frame.copy().astype(object)
    for col in frame.columns:
        shown[col] = [_fmt(v, col in pct_cols, col in signed) for v in frame[col]]
    print(shown.to_string())
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    from powerbacktest.engine.runner import run_backtest
    from powerbacktest.report.render import write_outputs

    extra = {"report.html": False} if args.no_html else {}
    config = _load(args, extra)
    with _data_manager(config) as data:
        result = run_backtest(config, data)
    out = write_outputs(result, args.out)
    print_summary(result)
    report = out / "report.html"
    print(f"\nReport: {report if report.exists() else out}")
    if args.open and report.exists():
        webbrowser.open(report.resolve().as_uri())
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from powerbacktest.engine.runner import load_data
    from powerbacktest.optimize import optimize, parse_grid_spec, summarize_table
    from powerbacktest.strategy.registry import create_strategy, load_strategy_paths

    extra: dict[str, Any] = {}
    for spec in args.grid:
        name, values = parse_grid_spec(spec)
        extra[f"optimize.grid.{name}"] = values
    for attr, dotted in (
        ("metric", "optimize.metric"),
        ("split", "optimize.split"),
        ("folds", "optimize.walk_forward_folds"),
        ("top", "optimize.top"),
        ("jobs", "optimize.jobs"),
        ("min_trades", "optimize.min_trades"),
    ):
        if getattr(args, attr) is not None:
            extra[dotted] = getattr(args, attr)
    config = _load(args, extra)
    if not config.optimize.grid:
        raise ConfigError(
            "Nothing to optimise: pass --grid name=start:stop:step or set optimize.grid"
        )
    if config.backtest.strategy_paths:
        load_strategy_paths(config.backtest.strategy_paths)
    warmup = 0
    for combo_values in _grid_corners(config.optimize.grid):
        try:
            strat = create_strategy(
                config.backtest.strategy, {**config.backtest.params, **combo_values}
            )
            warmup = max(warmup, strat.warmup_bars())
        except PowerBacktestError:
            continue
    with _data_manager(config) as data:
        loaded = load_data(config, data, warmup)
    result = optimize(config, loaded)
    out = (
        Path(args.out)
        if args.out
        else Path(config.report.output_dir) / f"optimize_{config.backtest.strategy}"
    )
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "optimize.csv", index=False)
    (out / "optimize.json").write_text(
        json.dumps(result.summary(), indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    if result.oos_equity is not None:
        result.oos_equity.to_csv(out / "walk_forward_equity.csv", index_label="time")
    print(summarize_table(result.table, config.optimize.top))
    summary = result.summary()
    if "walk_forward_oos_return" in summary:
        print(
            f"\nWalk-forward out-of-sample return: {summary['walk_forward_oos_return'] * 100:+.2f}%"
        )
    if result.skipped:
        print(
            f"{len(result.skipped)} parameter set(s) skipped (invalid or no data)", file=sys.stderr
        )
    print(f"\nResults: {out}")
    return 0


def _grid_corners(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Parameter sets at the extremes of each numeric range, to size the warm-up."""
    corners: list[dict[str, Any]] = [{}]
    for name, values in grid.items():
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        picks = [max(numeric)] if numeric else values[:1]
        corners = [{**c, name: p} for c in corners for p in picks]
    return corners


def cmd_data_fetch(args: argparse.Namespace) -> int:
    config = _load(args)
    bt = config.backtest
    tf = bt.tf
    with _data_manager(config) as data:
        instruments = data.instruments(bt.symbols, bt.lot_sizes)
        for sym in bt.symbols + ([bt.benchmark] if bt.benchmark else []):
            bars = data.load_bars(sym, tf, bt.start, bt.end, warmup_bars=args.warmup)
            first = bars.index[0].date() if len(bars) else "-"
            last = bars.index[-1].date() if len(bars) else "-"
            inst = instruments.get(sym)
            lot = f" lot {inst.lot_size}" if inst else ""
            print(
                f"{sym:<12} {tf.base!s:>5} {config.data.adjust:<4} {len(bars):>7} bars  {first} .. {last}{lot}"
            )
        for w in data.warnings:
            print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    store = ParquetStore(args.data_dir)
    rows = store.list_series()
    if not rows:
        print(f"No cached series under {store.root}")
        return 0
    frame = pd.DataFrame(rows)[
        [
            "symbol",
            "ktype",
            "adjust",
            "coverage",
            "rows",
            "first_bar",
            "last_bar",
            "source",
            "updated_at",
        ]
    ]
    frame["coverage"] = frame["coverage"].map(lambda c: f"{c[0]}..{c[1]}")
    print(frame.to_string(index=False))
    infos = store.read_instruments()
    if infos:
        print()
        print(
            pd.DataFrame([i.__dict__ for i in infos.values()])[
                ["symbol", "name", "lot_size", "security_type", "updated_at"]
            ].to_string(index=False)
        )
    return 0


def _store_import(
    store: ParquetStore, symbol: str, ktype: str, adjust: str, bars: pd.DataFrame, source: str
) -> None:
    if bars.empty:
        raise ConfigError("Nothing to import: no bars found")
    key = SeriesKey(symbol, ktype, adjust)
    existing = store.read(key)
    cov = store.coverage(key)
    tz = MARKETS[parse_symbol(symbol)[0]].tz
    dates = pd.DatetimeIndex(bars.index).tz_convert(tz).date
    new_cov = Coverage(min(dates), max(dates))
    if existing is not None and len(existing):
        bars = pd.concat([existing, bars])
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        if cov is not None:
            new_cov = cov.union(new_cov.start, new_cov.end)
    store.write(key, bars, new_cov, source)
    print(
        f"Imported {symbol} {ktype} {adjust}: {len(bars)} bars, coverage {new_cov.start}..{new_cov.end}"
    )


def cmd_data_import(args: argparse.Namespace) -> int:
    symbol = normalize_symbol(args.symbol)
    tf = Timeframe.parse(args.timeframe)
    if not tf.is_native:
        raise ConfigError(f"Import native timeframes only ({tf} is built by resampling {tf.base})")
    tz = MARKETS[parse_symbol(symbol)[0]].tz
    bars = read_bar_files(args.files, tz)
    _store_import(
        ParquetStore(args.data_dir),
        symbol,
        tf.futu_ktype,
        args.adjust,
        bars,
        f"import:{args.files[0]}",
    )
    return 0


def cmd_data_import_ticks(args: argparse.Namespace) -> int:
    symbol = normalize_symbol(args.symbol)
    market = MARKETS[parse_symbol(symbol)[0]]
    bars = ticks_to_minute_bars(args.files, symbol, market)
    _store_import(ParquetStore(args.data_dir), symbol, "K_1M", "none", bars, "futu_tick_downloader")
    return 0


def cmd_quota(args: argparse.Namespace) -> int:
    config = _load(
        args,
        {
            "backtest.symbols": ["HK.00700"],
            "backtest.start": "2000-01-01",
            "backtest.end": "2000-01-02",
        },
    )
    with FutuSource(config.futu.host, config.futu.port) as src:
        status = src.quota()
    if status is None:
        print("Quota unavailable (see log)")
        return 1
    print(f"History K-line quota: {status.used} used, {status.remaining} remaining")
    if status.codes:
        print("Symbols counted in the current window: " + ", ".join(sorted(status.codes)))
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    from powerbacktest.strategy.registry import available

    items = available()
    if args.json:
        print(
            json.dumps(
                {
                    n: {"title": c.title, "description": c.description, "params": c.param_schema()}
                    for n, c in items.items()
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    for name, cls in items.items():
        print(f"{name:<10} {cls.title}")
        print(f"{'':<10} {cls.description}")
        props = cls.param_schema().get("properties", {})
        for pname, spec in props.items():
            default = spec.get("default")
            print(
                f"{'':<12}{pname} = {default!r}"
                + (f"  ({spec['description']})" if spec.get("description") else "")
            )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from powerbacktest.report.render import rerender

    print(rerender(args.result, args.out))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if target.exists() and not args.force:
        raise ConfigError(f"{target} exists; use --force to overwrite")
    text = (resources.files("powerbacktest") / "example_config.yaml").read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"Wrote {target}")
    return 0


# -------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbt", description="strategy_powerbacktest: backtests on Futu OpenAPI data"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="Run a backtest and write the report")
    _add_run_options(p)
    p.add_argument("--out", help="Output directory (default: reports/<strategy>_<tf>_<timestamp>)")
    p.add_argument("--no-html", action="store_true", help="Skip report.html (JSON/CSV only)")
    p.add_argument("--open", action="store_true", help="Open the report in a browser")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("optimize", help="Grid search with in/out-of-sample validation")
    _add_run_options(p)
    p.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help="e.g. fast_period=5:20:5 or mode=cross,state",
    )
    p.add_argument("--metric", help="Metric to maximise (prefix '-' to minimise)")
    p.add_argument("--split", type=date.fromisoformat, help="Out-of-sample starts on this date")
    p.add_argument("--folds", type=int, help="Anchored walk-forward folds")
    p.add_argument("--top", type=int)
    p.add_argument("--jobs", type=int, help="Worker processes (0 = CPUs)")
    p.add_argument("--min-trades", type=int, dest="min_trades")
    p.add_argument("--out")
    p.set_defaults(func=cmd_optimize)

    data = sub.add_parser("data", help="Local data cache").add_subparsers(
        dest="data_command", required=True
    )
    p = data.add_parser("fetch", help="Download bars for the configured symbols into the cache")
    _add_run_options(p)
    p.add_argument("--warmup", type=int, default=0, help="Extra bars before start")
    p.set_defaults(func=cmd_data_fetch)
    p = data.add_parser("list", help="List cached series")
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_data_list)
    p = data.add_parser("import", help="Import CSV/Parquet bars (Futu, futu_algo or TDX exports)")
    p.add_argument("files", nargs="+")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="DAY")
    p.add_argument("--adjust", choices=["qfq", "hfq", "none"], required=True)
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_data_import)
    p = data.add_parser(
        "import-ticks", help="Build 1-minute bars from futu_tick_downloader SQLite files"
    )
    p.add_argument("files", nargs="+")
    p.add_argument("--symbol", required=True)
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_data_import_ticks)

    p = sub.add_parser("quota", help="Show the Futu history K-line quota")
    p.add_argument("-c", "--config")
    p.add_argument("--set", action="append", default=[])
    p.set_defaults(func=cmd_quota)

    p = sub.add_parser("strategies", help="List strategies and their parameters")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_strategies)

    p = sub.add_parser(
        "report", help="Re-render report.html from a result directory or result.json"
    )
    p.add_argument("result")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("init", help="Write an example config")
    p.add_argument("path", nargs="?", default="config.local.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in ("run", "optimize", "quota") and not (
        args.command == "data" and args.data_command == "fetch"
    ):
        setup_logging("INFO")
    try:
        return int(args.func(args) or 0)
    except PowerBacktestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # output piped into e.g. `head`
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
