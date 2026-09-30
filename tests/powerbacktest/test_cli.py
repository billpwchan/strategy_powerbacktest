from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from helpers import FakeQuoteContext, make_daily_bars

from powerbacktest.cli import main
from powerbacktest.config import load_config
from powerbacktest.data.manager import DataManager
from powerbacktest.data.source import FutuSource
from powerbacktest.data.store import Coverage, InstrumentInfo, ParquetStore, SeriesKey
from powerbacktest.engine.runner import run_backtest
from powerbacktest.optimize import expand_grid, fold_bounds, parse_grid_spec
from powerbacktest.report.render import write_outputs

ROOT = Path(__file__).resolve().parents[2]


def _seed_cache(root: Path) -> None:
    store = ParquetStore(root)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    infos = {}
    for i, (sym, lot, price) in enumerate([("HK.00700", 100, 300.0), ("HK.09988", 100, 90.0)]):
        bars = make_daily_bars(n=800, start="2021-06-01", seed=i + 1, price=price)
        cov = Coverage(bars.index[0].date(), bars.index[-1].date())
        store.write(SeriesKey(sym, "K_DAY", "qfq"), bars, cov, "test")
        infos[sym] = InstrumentInfo(sym, f"NAME{i}", lot, "STOCK", None, stamp)
    store.write_instruments(infos)


def _config(tmp_path: Path, **extra) -> Path:
    data = {
        "data": {"dir": str(tmp_path / "data"), "offline": True},
        "backtest": {
            "strategy": "macd",
            "symbols": ["HK.00700", "HK.09988"],
            "start": "2022-01-01",
            "end": "2024-06-30",
        },
        "report": {"output_dir": str(tmp_path / "reports")},
        "logging": {"level": "WARNING"},
        **extra,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_example_configs_are_identical_and_valid():
    packaged = (ROOT / "src/powerbacktest/example_config.yaml").read_text()
    assert packaged == (ROOT / "configs/example.yaml").read_text()
    cfg = load_config(ROOT / "configs/example.yaml")
    assert cfg.backtest.strategy == "macd"


def test_cli_run_report_and_rerender(tmp_path, capsys):
    _seed_cache(tmp_path / "data")
    cfg = _config(tmp_path)
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "-c",
            str(cfg),
            "--out",
            str(out),
            "-p",
            "fast_period=10",
            "--set",
            "risk.stop_loss=0.1",
        ]
    )
    assert code == 0
    for name in (
        "report.html",
        "result.json",
        "summary.csv",
        "equity.csv",
        "trades.csv",
        "fills.csv",
        "config.yaml",
    ):
        assert (out / name).exists(), name
    payload = json.loads((out / "result.json").read_text())
    assert payload["meta"]["strategy"]["params"]["fast_period"] == 10
    assert payload["meta"]["risk"]["stop_loss"] == 0.1
    assert payload["primary"] == "COMPOSITE" and payload["lookahead"]["ok"]
    html = (out / "report.html").read_text()
    assert "LightweightCharts" in html and "</script>" in html
    assert "COMPOSITE" in capsys.readouterr().out
    (out / "report.html").unlink()
    assert main(["report", str(out)]) == 0
    assert (out / "report.html").exists()


def test_cli_errors_return_code_2(tmp_path, capsys):
    _seed_cache(tmp_path / "data")
    cfg = _config(tmp_path)
    assert main(["run", "-c", str(cfg), "--strategy", "does_not_exist"]) == 2
    assert "Unknown strategy" in capsys.readouterr().err
    assert main(["run", "-c", str(cfg), "--symbols", "HK.00001"]) == 2  # not cached, offline


def test_cli_optimize_split_and_walk_forward(tmp_path, capsys):
    _seed_cache(tmp_path / "data")
    cfg = _config(tmp_path)
    out = tmp_path / "opt"
    code = main(
        [
            "optimize",
            "-c",
            str(cfg),
            "--grid",
            "fast_period=8:12:2",
            "--grid",
            "slow_period=20,26",
            "--split",
            "2023-07-01",
            "--jobs",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    import pandas as pd

    table = pd.read_csv(out / "optimize.csv")
    assert len(table) == 6 and "oos_sharpe" in table.columns and "is_sharpe" in table.columns
    code = main(
        [
            "optimize",
            "-c",
            str(cfg),
            "--grid",
            "fast_period=8,12",
            "--folds",
            "2",
            "--jobs",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    summary = json.loads((out / "optimize.json").read_text())
    assert summary["mode"] == "walk_forward" and len(summary["folds"]) == 2
    assert "walk_forward_oos_return" in summary


def test_grid_parsing_and_folds():
    assert parse_grid_spec("fast=5:20:5") == ("fast", [5, 10, 15, 20])
    assert parse_grid_spec("pct=0.5:1.5:0.5") == ("pct", [0.5, 1.0, 1.5])
    assert parse_grid_spec("mode=cross,state") == ("mode", ["cross", "state"])
    assert len(expand_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6
    folds = fold_bounds(date(2020, 1, 1), date(2023, 12, 31), 3)
    assert folds[0][0] == date(2020, 1, 1)
    assert all(tr_e < te_s <= te_e for _, tr_e, te_s, te_e in folds)
    assert folds[-1][3] == date(2023, 12, 31)


def test_cli_data_import_list_and_init(tmp_path, capsys):
    csv = tmp_path / "bars.csv"
    bars = make_daily_bars(n=30)
    frame = bars.reset_index()
    frame["time_key"] = frame["time"].dt.strftime("%Y-%m-%d 00:00:00")
    frame.drop(columns=["time"]).to_csv(csv, index=False)
    data_dir = tmp_path / "data"
    assert (
        main(
            [
                "data",
                "import",
                str(csv),
                "--symbol",
                "HK.700",
                "--adjust",
                "qfq",
                "--data-dir",
                str(data_dir),
            ]
        )
        == 0
    )
    cached = ParquetStore(data_dir).read(SeriesKey("HK.00700", "K_DAY", "qfq"))
    assert cached is not None and len(cached) == 30
    assert main(["data", "list", "--data-dir", str(data_dir)]) == 0
    assert "HK.00700" in capsys.readouterr().out
    target = tmp_path / "new.yaml"
    assert main(["init", str(target)]) == 0
    assert load_config(target).backtest.symbols
    assert main(["init", str(target)]) == 2


def test_full_pipeline_through_fake_opend(tmp_path):
    series = {
        ("HK.00700", "K_DAY", "qfq"): make_daily_bars(n=900, start="2021-06-01", seed=1, price=300),
        ("HK.09988", "K_DAY", "qfq"): make_daily_bars(n=900, start="2021-06-01", seed=2, price=90),
        ("HK.800000", "K_DAY", "qfq"): make_daily_bars(
            n=900, start="2021-06-01", seed=3, price=20000
        ),
    }
    ctx = FakeQuoteContext(series, {"HK.00700": 100, "HK.09988": 100})
    source = FutuSource(context_factory=lambda h, p: ctx, sleep=lambda s: None)
    data = DataManager(
        ParquetStore(tmp_path / "data"), lambda: source, today=lambda tz: date(2026, 9, 30)
    )
    cfg = load_config(
        None,
        {
            "backtest.strategy": "btse",
            "backtest.symbols": ["HK.00700", "HK.09988"],
            "backtest.start": "2022-01-01",
            "backtest.end": "2024-12-31",
            "backtest.benchmark": "HK.800000",
            "backtest.mode": "portfolio",
            "report.output_dir": str(tmp_path / "reports"),
        },
    )
    result = run_backtest(cfg, data)
    assert (
        result.main.benchmark_name == "NAME HK.800000" or result.main.benchmark_name == "HK.800000"
    )
    assert "beta" in result.main.metrics
    out = write_outputs(result)
    assert (out / "report.html").exists()
    n_calls = len(ctx.calls)
    run_backtest(cfg, data)  # second run is served entirely from cache
    assert [c for c in ctx.calls[n_calls:] if c[0] == "kline"] == []
