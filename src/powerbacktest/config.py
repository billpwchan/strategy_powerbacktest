"""Run configuration.

One YAML file (see ``configs/example.yaml``) validated by these models is the only place
settings live; the CLI applies overrides on top with dotted ``--set key=value`` paths.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from powerbacktest.errors import ConfigError
from powerbacktest.market.costs import HKBrokerFees, USBrokerFees
from powerbacktest.market.instrument import normalize_symbol
from powerbacktest.timeframe import Timeframe

Adjust = Literal["qfq", "hfq", "none"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FutuConfig(_Model):
    host: str = "127.0.0.1"
    port: int = 11111
    # Requests per window; Futu allows 60 history K-line requests per 30 seconds.
    rate_limit: int = Field(60, ge=1)
    rate_window_seconds: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0)


class DataConfig(_Model):
    dir: Path = Path("data")
    adjust: Adjust = "qfq"
    offline: bool = False
    instrument_ttl_days: int = Field(7, ge=0)


class BacktestConfig(_Model):
    strategy: str = "macd"
    params: dict[str, Any] = Field(default_factory=dict)
    symbols: list[str] = Field(default_factory=list)
    start: dt.date
    end: dt.date
    timeframe: str = "DAY"
    mode: Literal["scan", "portfolio"] = "scan"
    initial_capital: float = Field(100_000.0, gt=0)
    benchmark: str | None = None
    lot_sizes: dict[str, int] = Field(default_factory=dict)
    strategy_paths: list[Path] = Field(default_factory=list)
    # Recompute the strategy on truncated history to detect future functions.
    check_lookahead: bool = True

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, value: list[str]) -> list[str]:
        normalized = [normalize_symbol(s) for s in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate symbols")
        return normalized

    @field_validator("benchmark")
    @classmethod
    def _benchmark(cls, value: str | None) -> str | None:
        return normalize_symbol(value) if value else None

    @field_validator("lot_sizes")
    @classmethod
    def _lots(cls, value: dict[str, int]) -> dict[str, int]:
        return {normalize_symbol(k): int(v) for k, v in value.items()}

    @field_validator("timeframe")
    @classmethod
    def _timeframe(cls, value: str) -> str:
        return str(Timeframe.parse(value))

    @model_validator(mode="after")
    def _dates(self) -> BacktestConfig:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self

    @property
    def tf(self) -> Timeframe:
        return Timeframe.parse(self.timeframe)


class ExecutionConfig(_Model):
    # next_open: decide on bar t's close, fill at bar t+1's open (default, no look-ahead).
    # close: fill at the signal bar's close, as TDX's built-in backtest does (optimistic).
    fill: Literal["next_open", "close"] = "next_open"
    slippage_ticks: float = Field(1.0, ge=0)
    slippage_bps: float = Field(0.0, ge=0)
    max_volume_pct: float | None = Field(None, gt=0, le=1)
    liquidate_at_end: bool = False
    # After a stop/take-profit exit, and at the start of the test, require the strategy's
    # signal to leave the "long" state before entering again.
    entry_requires_fresh_signal: bool = True


class RiskConfig(_Model):
    stop_loss: float | None = Field(None, gt=0, lt=1)
    take_profit: float | None = Field(None, gt=0)
    trailing_stop: float | None = Field(None, gt=0, lt=1)
    max_holding_bars: int | None = Field(None, ge=1)


class PortfolioConfig(_Model):
    # auto: all_in in scan mode, equal_weight in portfolio mode.
    sizer: Literal[
        "auto", "all_in", "equal_weight", "percent_equity", "fixed_value", "fixed_lots"
    ] = "auto"
    max_positions: int = Field(5, ge=1)
    percent: float | None = Field(None, gt=0, le=1)
    value: float | None = Field(None, gt=0)
    lots: int = Field(1, ge=1)

    def resolved_sizer(self, mode: str) -> str:
        if self.sizer != "auto":
            return self.sizer
        return "all_in" if mode == "scan" else "equal_weight"


class SimpleCosts(_Model):
    rate: float = Field(0.001, ge=0)
    min_fee: float = Field(0.0, ge=0)


class CostsConfig(_Model):
    model: Literal["market", "simple"] = "market"
    hk: HKBrokerFees = Field(default_factory=HKBrokerFees)
    us: USBrokerFees = Field(default_factory=USBrokerFees)
    simple: SimpleCosts = Field(default_factory=SimpleCosts)


class AnalyticsConfig(_Model):
    risk_free_rate: float = 0.0


class ReportConfig(_Model):
    output_dir: Path = Path("reports")
    title: str | None = None
    html: bool = True
    max_bars_per_symbol: int = Field(20_000, ge=100)


class LoggingConfig(_Model):
    level: str = "INFO"
    file: Path | None = None


class OptimizeConfig(_Model):
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    # Metric of the primary book to maximise; prefix with "-" to minimise (e.g. "-volatility").
    metric: str = "sharpe"
    # In-sample ends the day before ``split``; out-of-sample runs from ``split`` to ``end``.
    split: dt.date | None = None
    # Anchored walk-forward: the period is cut into folds + 1 equal slices; fold i trains on
    # slices 0..i and tests on slice i + 1.
    walk_forward_folds: int | None = Field(None, ge=2)
    top: int = Field(10, ge=1)
    jobs: int = Field(0, ge=0, description="0 = one worker per CPU")
    min_trades: int = Field(0, ge=0, description="Ignore parameter sets with fewer closed trades")


class RunConfig(_Model):
    futu: FutuConfig = Field(default_factory=FutuConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    optimize: OptimizeConfig = Field(default_factory=OptimizeConfig)

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if not self.backtest.symbols:
            raise ValueError("backtest.symbols must list at least one symbol")
        if self.backtest.mode == "portfolio":
            currencies = {s.split(".")[0] for s in self.backtest.symbols}
            if len(currencies) > 1:
                raise ValueError(
                    "portfolio mode shares one cash balance and needs a single market/currency; "
                    f"got markets {sorted(currencies)}. Use mode: scan for mixed markets."
                )
        sizer = self.portfolio.resolved_sizer(self.backtest.mode)
        if sizer == "percent_equity" and self.portfolio.percent is None:
            raise ValueError("portfolio.percent is required for sizer percent_equity")
        if sizer == "fixed_value" and self.portfolio.value is None:
            raise ValueError("portfolio.value is required for sizer fixed_value")
        return self


def set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``data['a']['b']['c'] = value`` for ``dotted='a.b.c'``, creating dicts on the way."""
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` where value is YAML (so numbers, lists and bools work)."""
    if "=" not in text:
        raise ConfigError(f"Override {text!r} must look like key=value")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw) if raw.strip() else None


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config file {p} must contain a mapping at the top level")
        data = loaded
    for key, value in (overrides or {}).items():
        set_dotted(data, key, value)
    try:
        return RunConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def dump_config(config: RunConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
