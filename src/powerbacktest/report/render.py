"""Write a backtest's outputs: a self-contained HTML report plus machine-readable files.

Output directory layout::

    report.html     single file, works offline (chart library inlined)
    result.json     everything the report shows; ``pbt report`` re-renders from it
    summary.csv     headline metrics per book
    equity.csv      equity curve per book
    trades.csv      round trips
    fills.csv       individual executions with fee breakdown
    config.yaml     the fully resolved configuration
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, select_autoescape

from powerbacktest.config import dump_config
from powerbacktest.engine.runner import BacktestResult
from powerbacktest.report.payload import build_payload


def _asset(name: str) -> str:
    return (resources.files("powerbacktest.report") / name).read_text(encoding="utf-8")


def render_html(payload: dict[str, Any]) -> str:
    env = Environment(autoescape=select_autoescape(default=True))
    template = env.from_string(_asset("templates/report.html.j2"))
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Keep the JSON inert inside <script>: no "</script>" or "<!--" sequences.
    payload_json = (
        payload_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    return template.render(
        title=payload["meta"]["title"],
        chart_js=_asset("static/lightweight-charts.standalone.production.js"),
        payload_json=payload_json,
    )


def default_output_dir(result: BacktestResult) -> Path:
    base = Path(result.config.report.output_dir)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result.strategy["name"]))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"{name}_{result.config.backtest.timeframe}_{stamp}"


def write_outputs(result: BacktestResult, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else default_output_dir(result)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(result)
    (out / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8"
    )
    if result.config.report.html:
        (out / "report.html").write_text(render_html(payload), encoding="utf-8")
    (out / "config.yaml").write_text(dump_config(result.config), encoding="utf-8")

    result.summary().to_csv(out / "summary.csv", index_label="book")
    equity = pd.concat({name: b.equity for name, b in result.books.items()}, axis=1)
    equity.to_csv(out / "equity.csv", index_label="time")
    trades = [
        dict(book=name, **t.to_dict())
        for name, b in result.books.items()
        if name != "COMPOSITE"
        for t in b.trades
    ]
    pd.DataFrame(trades).to_csv(out / "trades.csv", index=False)
    fills = [
        dict(book=name, **f.to_dict())
        for name, b in result.books.items()
        if name != "COMPOSITE"
        for f in b.fills
    ]
    pd.DataFrame(fills).to_csv(out / "fills.csv", index=False)
    return out


def rerender(result_json: str | Path, out_html: str | Path | None = None) -> Path:
    src = Path(result_json)
    if src.is_dir():
        src = src / "result.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    target = Path(out_html) if out_html else src.with_name("report.html")
    target.write_text(render_html(payload), encoding="utf-8")
    return target
