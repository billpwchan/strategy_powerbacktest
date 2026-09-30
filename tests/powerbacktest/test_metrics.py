from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from powerbacktest.analytics.metrics import (
    bar_returns,
    benchmark_metrics,
    drawdown_periods,
    monthly_returns,
    periods_per_year,
    return_metrics,
    trade_stats,
)
from powerbacktest.engine.types import Trade

TZ = "Asia/Hong_Kong"


def _eq(values, start="2023-01-02", freq="B"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq=freq, tz=TZ), dtype=float
    )


def test_drawdown_duration_counts_the_longest_underwater_stretch():
    # up, 3-bar drawdown and recovery, then a 5-bar drawdown that never recovers
    eq = _eq([100, 110, 105, 104, 103, 111, 112, 108, 107, 106, 105, 104])
    periods = drawdown_periods(eq, 100.0, top=5)
    worst_by_len = max(periods, key=lambda p: p["bars"])
    assert worst_by_len["bars"] == 5 and not worst_by_len["recovered"]
    first = next(p for p in periods if p["recovered"])
    assert first["bars"] == 4  # peak at 110 (bar 1) to recovery at bar 5
    assert min(p["depth"] for p in periods) == pytest.approx(104 / 112 - 1)


def test_monthly_returns_are_labelled_with_their_own_month():
    idx = pd.bdate_range("2023-01-02", "2023-03-31", tz=TZ)
    eq = pd.Series(100.0, index=idx)
    eq[idx.month >= 2] = 110.0  # +10% happens in February
    eq[idx.month == 3] = 110.0
    table = monthly_returns(eq, 100.0)
    assert table.loc[2023, 1] == pytest.approx(0.0)
    assert table.loc[2023, 2] == pytest.approx(0.10)
    assert table.loc[2023, 3] == pytest.approx(0.0)
    assert table.loc[2023, "year"] == pytest.approx(0.10)


def test_return_metrics_match_definitions():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 504)
    eq = _eq(100_000 * np.cumprod(1 + r))
    m = return_metrics(eq, 100_000.0, fallback_ppy=252, risk_free_rate=0.02)
    rets = bar_returns(eq, 100_000.0)
    ppy = periods_per_year(pd.DatetimeIndex(eq.index), 252)
    assert ppy == pytest.approx(503 / ((eq.index[-1] - eq.index[0]).days / 365.25))
    assert m["sharpe"] == pytest.approx(
        (rets.mean() - 0.02 / ppy) / rets.std(ddof=1) * math.sqrt(ppy)
    )
    downside = np.sqrt(np.mean(np.minimum(rets - 0.02 / ppy, 0) ** 2))
    assert m["sortino"] == pytest.approx(
        (rets.mean() - 0.02 / ppy) * ppy / (downside * math.sqrt(ppy))
    )
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    assert m["cagr"] == pytest.approx((eq.iloc[-1] / 100_000) ** (1 / years) - 1, rel=1e-4)
    assert m["total_return"] == pytest.approx(eq.iloc[-1] / 100_000 - 1)


def test_intraday_annualisation_uses_bar_count():
    idx = pd.date_range("2023-01-02 10:30", periods=6 * 250 * 2, freq="h", tz=TZ)
    ppy = periods_per_year(pd.DatetimeIndex(idx), 252)
    assert ppy > 5000  # hourly bars, not 252


def test_benchmark_beta_of_itself_is_one():
    eq = _eq(np.linspace(100, 130, 300) + np.sin(np.arange(300)) * 2)
    m = benchmark_metrics(eq, eq, 100.0, fallback_ppy=252)
    assert m["beta"] == pytest.approx(1.0) and m["correlation"] == pytest.approx(1.0)
    assert m["excess_return"] == pytest.approx(0.0)


def test_trade_stats():
    t0 = pd.Timestamp("2024-01-02", tz=TZ)

    def trade(pnl, bars):
        t = Trade("HK.00700", t0, 0, 100, 10_000.0, 0.0)
        t.exit_value, t.exit_index, t.exit_time, t.exit_reason = (
            10_000.0 + pnl,
            bars,
            t0 + pd.Timedelta(days=bars),
            "signal",
        )
        return t

    trades = [trade(100, 2), trade(-50, 3), trade(-25, 1), trade(200, 4)]
    s = trade_stats(trades)
    assert s["trades"] == 4 and s["win_rate"] == 0.5
    assert s["profit_factor"] == pytest.approx(300 / 75)
    assert s["max_consecutive_losses"] == 2
    assert s["avg_bars_held"] == pytest.approx(2.5)
    assert s["payoff_ratio"] == pytest.approx(150 / 37.5)
