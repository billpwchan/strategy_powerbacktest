from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest
from helpers import FakeQuoteContext, make_daily_bars

from powerbacktest.data.importers import minute_labels, read_bar_files, ticks_to_minute_bars
from powerbacktest.data.manager import DataManager, last_complete_date
from powerbacktest.data.resample import resample_bars, session_labels
from powerbacktest.data.schema import to_bars, validate_bars
from powerbacktest.data.source import FutuSource, RateLimiter
from powerbacktest.data.store import ParquetStore, SeriesKey
from powerbacktest.errors import DataError, DataSourceError, QuotaExceededError
from powerbacktest.market.instrument import MARKETS
from powerbacktest.timeframe import Timeframe

HK = MARKETS["HK"]


def _manager(tmp_path, ctx, *, today=date(2026, 9, 30), offline=False):
    store = ParquetStore(tmp_path)
    source = FutuSource(context_factory=lambda h, p: ctx, sleep=lambda s: None)
    return DataManager(store, lambda: source, offline=offline, today=lambda tz: today)


def _full_series():
    return make_daily_bars(n=900, start="2022-01-03")


# ---------------------------------------------------------------- source


def test_rate_limiter_waits_when_window_full():
    clock = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        clock[0] += s

    limiter = RateLimiter(2, 30.0, clock=lambda: clock[0], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert slept and slept[0] == pytest.approx(30.05)


def test_futu_source_paginates_and_normalizes():
    bars = _full_series()
    ctx = FakeQuoteContext({("HK.00700", "K_DAY", "qfq"): bars}, {"HK.00700": 100})
    src = FutuSource(context_factory=lambda h, p: ctx, sleep=lambda s: None)
    out = src.fetch_bars("HK.00700", "K_DAY", "qfq", date(2022, 1, 1), date(2026, 1, 1))
    pages = [c for c in ctx.calls if c[0] == "kline"]
    assert len(pages) == 1 + (len(bars) - 1) // 1000
    assert len(out) == len(bars)
    assert str(out.index.tz) == "Asia/Hong_Kong"
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "turnover"]


def test_futu_source_retries_rate_limit_then_raises_on_other_errors():
    bars = _full_series()
    ctx = FakeQuoteContext({("HK.00700", "K_DAY", "qfq"): bars}, {})
    ctx.fail_next = [(-1, "获取历史K线频率太高")]
    src = FutuSource(context_factory=lambda h, p: ctx, sleep=lambda s: None)
    assert len(src.fetch_bars("HK.00700", "K_DAY", "qfq", date(2022, 1, 1), date(2022, 3, 1))) > 0
    ctx.fail_next = [(-1, "unknown stock")]
    with pytest.raises(DataSourceError):
        src.fetch_bars("HK.00700", "K_DAY", "qfq", date(2022, 1, 1), date(2022, 3, 1))
    ctx.fail_next = [(-1, "历史K线额度不足")]
    with pytest.raises(QuotaExceededError):
        src.fetch_bars("HK.00700", "K_DAY", "qfq", date(2022, 1, 1), date(2022, 3, 1))


# ---------------------------------------------------------------- cache


def test_manager_caches_and_only_fetches_missing_ranges(tmp_path):
    bars = _full_series()
    ctx = FakeQuoteContext({("HK.00700", "K_DAY", "qfq"): bars}, {"HK.00700": 100})
    dm = _manager(tmp_path, ctx)
    key = SeriesKey("HK.00700", "K_DAY", "qfq")
    first = dm.ensure(key, date(2023, 1, 1), date(2023, 6, 30))
    assert first.index[0].date() >= date(2023, 1, 1)
    n_calls = len(ctx.calls)
    again = dm.ensure(key, date(2023, 2, 1), date(2023, 5, 31))
    assert len(ctx.calls) == n_calls, "fully cached range must not hit the source"
    assert len(again) < len(first)
    dm.ensure(key, date(2022, 6, 1), date(2023, 9, 30))
    ranges = [(c[2], c[3]) for c in ctx.calls[n_calls:] if c[0] == "kline"]
    assert ranges == [("2022-06-01", "2023-01-11"), ("2023-06-20", "2023-09-30")]
    cov = dm.store.coverage(key)
    assert (cov.start, cov.end) == (date(2022, 6, 1), date(2023, 9, 30))


def test_manager_never_caches_in_progress_period(tmp_path):
    bars = _full_series()
    ctx = FakeQuoteContext({("HK.00700", "K_DAY", "qfq"): bars}, {})
    dm = _manager(tmp_path, ctx, today=date(2023, 3, 15))
    dm.ensure(SeriesKey("HK.00700", "K_DAY", "qfq"), date(2023, 1, 1), date(2023, 12, 31))
    assert dm.store.coverage(SeriesKey("HK.00700", "K_DAY", "qfq")).end == date(2023, 3, 14)
    assert last_complete_date("K_WEEK", date(2026, 9, 30)) == date(2026, 9, 27)
    assert last_complete_date("K_MON", date(2026, 9, 30)) == date(2026, 8, 31)


def test_manager_refetches_everything_when_adjustment_changes(tmp_path):
    bars = _full_series()
    ctx = FakeQuoteContext({("HK.00700", "K_DAY", "qfq"): bars}, {})
    dm = _manager(tmp_path, ctx)
    key = SeriesKey("HK.00700", "K_DAY", "qfq")
    dm.ensure(key, date(2022, 3, 1), date(2022, 12, 31))
    # A dividend after the cached range rescales every earlier forward-adjusted price.
    ctx.series[("HK.00700", "K_DAY", "qfq")] = bars * [0.97, 0.97, 0.97, 0.97, 1, 1]
    out = dm.ensure(key, date(2022, 3, 1), date(2023, 6, 30))
    assert any("re-downloaded" in w for w in dm.warnings)
    expected = ctx.series[("HK.00700", "K_DAY", "qfq")]
    np.testing.assert_allclose(out["close"].to_numpy(), expected.loc[out.index, "close"].to_numpy())


def test_offline_mode_uses_cache_and_errors_when_missing(tmp_path):
    bars = _full_series()
    ctx = FakeQuoteContext({("HK.00700", "K_DAY", "qfq"): bars}, {"HK.00700": 100})
    _manager(tmp_path, ctx).ensure(
        SeriesKey("HK.00700", "K_DAY", "qfq"), date(2023, 1, 1), date(2023, 6, 30)
    )
    offline = DataManager(ParquetStore(tmp_path), None, today=lambda tz: date(2026, 9, 30))
    out = offline.ensure(
        SeriesKey("HK.00700", "K_DAY", "qfq"), date(2023, 1, 1), date(2023, 12, 31)
    )
    assert len(out) > 0 and offline.warnings
    with pytest.raises(DataError):
        offline.ensure(SeriesKey("HK.09988", "K_DAY", "qfq"), date(2023, 1, 1), date(2023, 6, 30))


def test_quota_guard_blocks_new_symbols_only(tmp_path):
    bars = _full_series()
    ctx = FakeQuoteContext(
        {("HK.00700", "K_DAY", "qfq"): bars, ("HK.09988", "K_DAY", "qfq"): bars}, {}
    )
    ctx.quota_remaining = 0
    ctx.quota_codes = ["HK.00700"]
    dm = _manager(tmp_path, ctx)
    dm.ensure(SeriesKey("HK.00700", "K_DAY", "qfq"), date(2023, 1, 1), date(2023, 2, 1))
    with pytest.raises(QuotaExceededError):
        dm.ensure(SeriesKey("HK.09988", "K_DAY", "qfq"), date(2023, 1, 1), date(2023, 2, 1))


def test_instruments_fetch_cache_and_override(tmp_path):
    ctx = FakeQuoteContext({}, {"HK.00700": 100, "HK.09988": 50})
    dm = _manager(tmp_path, ctx)
    inst = dm.instruments(["HK.00700", "HK.09988", "US.AAPL"], {"HK.09988": 100})
    assert inst["HK.00700"].lot_size == 100
    assert inst["HK.09988"].lot_size == 100
    assert inst["US.AAPL"].lot_size == 1
    offline = DataManager(ParquetStore(tmp_path), None)
    assert offline.instruments(["HK.00700"])["HK.00700"].name == "NAME HK.00700"
    with pytest.raises(DataError):
        offline.instruments(["HK.00005"])


def test_load_bars_includes_warmup_and_resamples(tmp_path):
    idx = []
    for day in pd.bdate_range("2024-03-01", periods=60):
        for hm in ["10:30", "11:30", "12:00", "14:00", "15:00", "16:00"]:
            idx.append(pd.Timestamp(f"{day.date()} {hm}"))
    n = len(idx)
    raw = pd.DataFrame(
        {
            "time_key": idx,
            "open": np.arange(n) + 1.0,
            "high": np.arange(n) + 2.0,
            "low": np.arange(n) + 0.5,
            "close": np.arange(n) + 1.5,
            "volume": 10.0,
            "turnover": 1.0,
        }
    )
    bars = to_bars(raw, "Asia/Hong_Kong")
    ctx = FakeQuoteContext({("HK.00700", "K_60M", "qfq"): bars}, {})
    dm = _manager(tmp_path, ctx)
    out = dm.load_bars(
        "HK.00700", Timeframe.parse("4H"), date(2024, 5, 1), date(2024, 5, 20), warmup_bars=20
    )
    assert out.index[0].date() < date(2024, 5, 1)
    assert {t.strftime("%H:%M") for t in out.index} == {"12:00", "16:00"}


# ------------------------------------------------------------- resample


def test_hk_session_labels_for_2h_and_4h():
    day = "2024-03-04"
    idx = pd.DatetimeIndex(
        [f"{day} {t}" for t in ["10:30", "11:30", "12:00", "14:00", "15:00", "16:00"]],
        tz="Asia/Hong_Kong",
    )
    assert [t.strftime("%H:%M") for t in session_labels(idx, 120, HK)] == [
        "11:30",
        "11:30",
        "12:00",
        "15:00",
        "15:00",
        "16:00",
    ]
    assert [t.strftime("%H:%M") for t in session_labels(idx, 240, HK)] == [
        "12:00",
        "12:00",
        "12:00",
        "16:00",
        "16:00",
        "16:00",
    ]


def test_resample_aggregates_ohlcv_with_auction_bar():
    idx = pd.DatetimeIndex(
        ["2024-03-04 09:30", "2024-03-04 09:31", "2024-03-04 09:35", "2024-03-04 09:36"],
        tz="Asia/Hong_Kong",
    )
    bars = pd.DataFrame(
        {
            "open": [10, 11, 12, 13],
            "high": [10, 12, 14, 13.5],
            "low": [10, 10.5, 11, 12],
            "close": [10, 11.5, 13, 13.2],
            "volume": [5, 1, 2, 3],
            "turnover": [np.nan] * 4,
        },
        index=idx,
    ).astype(float)
    out = resample_bars(bars, Timeframe.parse("1M"), Timeframe.parse("5M"), HK)
    assert [t.strftime("%H:%M") for t in out.index] == ["09:35", "09:40"]
    first = out.iloc[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == (10, 14, 10, 13, 8)
    assert np.isnan(first.turnover)


def test_us_session_labels():
    idx = pd.DatetimeIndex(
        [
            f"2024-03-04 {t}"
            for t in ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
        ],
        tz="America/New_York",
    )
    labels = session_labels(idx, 120, MARKETS["US"])
    assert [t.strftime("%H:%M") for t in labels] == [
        "11:30",
        "11:30",
        "13:30",
        "13:30",
        "15:30",
        "15:30",
        "16:00",
    ]


# --------------------------------------------------------------- imports


def test_read_bar_files_accepts_tdx_chinese_headers(tmp_path):
    path = tmp_path / "tdx.csv"
    path.write_text(
        "日期,开盘,最高,最低,收盘,成交量,成交额\n2024-01-02,10,11,9.5,10.5,1000,10500\n",
        encoding="gbk",
    )
    bars = read_bar_files([path], "Asia/Hong_Kong")
    assert bars.iloc[0].close == 10.5 and str(bars.index.tz) == "Asia/Hong_Kong"


def test_ticks_to_minute_bars_matches_futu_labels(tmp_path):
    db = tmp_path / "20240304.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE ticks (symbol TEXT, ts_ms INTEGER, price REAL, volume INTEGER, turnover REAL, seq INTEGER)"
        )
        hk = "Asia/Hong_Kong"
        rows = [
            ("09:20:00", 10.0, 100),  # opening auction -> 09:30 bar
            ("09:30:05", 10.1, 200),  # -> 09:31
            ("09:30:59", 10.3, 50),  # -> 09:31
            ("11:59:59", 10.2, 10),  # -> 12:00
            ("16:08:30", 10.4, 500),  # closing auction -> 16:00
        ]
        for i, (t, p, v) in enumerate(rows):
            ts = int(pd.Timestamp(f"2024-03-04 {t}", tz=hk).tz_convert("UTC").value // 1_000_000)
            conn.execute("INSERT INTO ticks VALUES (?,?,?,?,?,?)", ("HK.00700", ts, p, v, p * v, i))
    bars = ticks_to_minute_bars([db], "HK.00700", HK)
    assert [t.strftime("%H:%M") for t in bars.index] == ["09:30", "09:31", "12:00", "16:00"]
    b931 = bars.iloc[1]
    assert (b931.open, b931.high, b931.close, b931.volume) == (10.1, 10.3, 10.3, 250)
    labels = minute_labels(np.array([ts for ts in []], dtype="int64"), HK)
    assert len(labels) == 0


def test_validate_bars_flags_bad_rows():
    bars = make_daily_bars(n=5)
    bars.iloc[2, bars.columns.get_loc("high")] = bars.iloc[2]["low"] - 1
    assert validate_bars(bars)
