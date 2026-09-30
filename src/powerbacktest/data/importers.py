"""Bring bars into the cache from files instead of OpenD.

* :func:`read_bar_files` loads CSV/Parquet exports: Futu-style columns (``time_key``, ``open``,
  ...), futu_algo's Parquet cache, or TDX exports with Chinese headers (日期, 开盘, 最高, ...).
* :func:`ticks_to_minute_bars` aggregates the trade ticks recorded by ``futu_tick_downloader``
  (one SQLite file per trading day, ``ticks`` table, UTC epoch-millisecond timestamps) into
  1-minute bars labelled the way Futu labels them, so they line up with fetched K-lines.
"""

from __future__ import annotations

import glob
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from powerbacktest.data.schema import BAR_COLUMNS, empty_bars, to_bars
from powerbacktest.errors import DataError
from powerbacktest.market.instrument import MarketSpec


def expand_paths(patterns: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if not matches and Path(pattern).exists():
            matches = [str(pattern)]
        paths.extend(Path(m) for m in matches)
    if not paths:
        raise DataError(f"No files matched {list(patterns)}")
    return paths


def read_bar_files(patterns: Iterable[str | Path], tz: str) -> pd.DataFrame:
    frames = []
    for path in expand_paths(patterns):
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif suffix in {".csv", ".txt"}:
            raw = _read_csv(path)
        else:
            raise DataError(f"Unsupported file type {path.suffix!r} for {path}")
        frames.append(to_bars(raw, tz))
    bars = pd.concat(frames)
    return bars[~bars.index.duplicated(keep="last")].sort_index()


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gbk", "big5"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DataError(f"Could not decode {path} as UTF-8, GBK or Big5")


def minute_labels(ts_ms: np.ndarray, market: MarketSpec) -> pd.DatetimeIndex:
    """Futu-style end labels for trades: a trade at 10:15:30 belongs to the 10:16 bar.

    Trades before the first session (opening auction) map to the session open bar, trades in
    a break map to the preceding session's last bar, and trades after the close (closing
    auction) map to the final bar.
    """
    utc = pd.to_datetime(ts_ms, unit="ms", utc=True)
    local = pd.DatetimeIndex(utc).tz_convert(market.tz)
    label = local.floor("min") + pd.Timedelta(minutes=1)
    minute = (label.hour * 60 + label.minute).to_numpy()
    day = label.normalize()
    starts = [s.hour * 60 + s.minute for s, _ in market.sessions]
    ends = [e.hour * 60 + e.minute for _, e in market.sessions]
    fixed = minute.copy()
    fixed = np.where(minute <= starts[0], starts[0], fixed)
    for i in range(len(starts) - 1):
        in_break = (minute > ends[i]) & (minute <= starts[i + 1])
        fixed = np.where(in_break, ends[i], fixed)
    fixed = np.where(minute > ends[-1], ends[-1], fixed)
    return pd.DatetimeIndex(day + pd.to_timedelta(fixed, unit="min"), name="time")


def ticks_to_minute_bars(
    db_patterns: Iterable[str | Path], symbol: str, market: MarketSpec
) -> pd.DataFrame:
    frames = []
    for path in expand_paths(db_patterns):
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            frames.append(
                pd.read_sql_query(
                    "SELECT ts_ms, price, volume, turnover FROM ticks "
                    "WHERE symbol = ? AND price IS NOT NULL ORDER BY ts_ms, seq",
                    conn,
                    params=(symbol,),
                )
            )
    ticks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if ticks.empty:
        return empty_bars(market.tz)
    ticks = ticks.sort_values("ts_ms", kind="stable")
    labels = minute_labels(ticks["ts_ms"].to_numpy(dtype="int64"), market)
    grouped = ticks.groupby(labels, sort=True)
    bars = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["volume"].sum().astype("float64"),
            "turnover": grouped["turnover"].sum(min_count=1),
        }
    )
    bars.index.name = "time"
    return bars.loc[:, list(BAR_COLUMNS)].astype("float64")
