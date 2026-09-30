from __future__ import annotations

import numpy as np
import pandas as pd

from powerbacktest.data.schema import to_bars


def make_daily_bars(
    n: int = 300,
    start: str = "2022-01-03",
    tz: str = "Asia/Hong_Kong",
    seed: int = 7,
    drift: float = 0.0005,
    vol: float = 0.02,
    price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    close = price * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    frame = pd.DataFrame(
        {
            "time_key": days.strftime("%Y-%m-%d 00:00:00"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
            "turnover": close * 500_000,
        }
    )
    return to_bars(frame, tz)


def bars_from_prices(
    opens: list[float],
    closes: list[float] | None = None,
    start: str = "2024-01-02",
    tz: str = "Asia/Hong_Kong",
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volume: float = 1e9,
) -> pd.DataFrame:
    closes = closes if closes is not None else opens
    days = pd.bdate_range(start, periods=len(opens))
    highs = highs if highs is not None else [max(o, c) for o, c in zip(opens, closes, strict=True)]
    lows = lows if lows is not None else [min(o, c) for o, c in zip(opens, closes, strict=True)]
    frame = pd.DataFrame(
        {
            "time": days,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "turnover": np.nan,
        }
    )
    return to_bars(frame, tz)


class FakeQuoteContext:
    """Mimics the parts of futu.OpenQuoteContext the package uses."""

    def __init__(self, series: dict[tuple[str, str, str], pd.DataFrame], lot_sizes: dict[str, int]):
        self.series = series
        self.lot_sizes = lot_sizes
        self.calls: list[tuple] = []
        self.closed = False
        self.quota_used = 0
        self.quota_remaining = 1000
        self.quota_codes: list[str] = []
        self.fail_next: list[tuple[int, str]] = []

    def request_history_kline(
        self,
        code,
        start=None,
        end=None,
        ktype="K_DAY",
        autype="qfq",
        max_count=1000,
        page_req_key=None,
        **_,
    ):
        self.calls.append(("kline", code, start, end, ktype, autype, page_req_key))
        if self.fail_next:
            ret, msg = self.fail_next.pop(0)
            return ret, msg, None
        adjust = {"qfq": "qfq", "hfq": "hfq", "None": "none"}[autype]
        bars = self.series.get((code, ktype, adjust))
        if bars is None:
            return 0, pd.DataFrame(), None
        local = bars.index.tz_localize(None)
        mask = (local.normalize() >= pd.Timestamp(start)) & (local.normalize() <= pd.Timestamp(end))
        sel = bars.loc[mask]
        offset = int(page_req_key or 0)
        page = sel.iloc[offset : offset + max_count]
        nxt = offset + max_count
        frame = page.reset_index()
        frame["time_key"] = frame["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        frame = frame.drop(columns=["time"])
        frame.insert(0, "code", code)
        return 0, frame, (str(nxt).encode() if nxt < len(sel) else None)

    def get_stock_basicinfo(self, market, stock_type="STOCK", code_list=None):
        self.calls.append(("basicinfo", tuple(code_list or [])))
        rows = [
            {
                "code": c,
                "name": f"NAME {c}",
                "lot_size": self.lot_sizes[c],
                "stock_type": "STOCK",
                "listing_date": "2004-06-16",
            }
            for c in code_list or []
            if c in self.lot_sizes
        ]
        return 0, pd.DataFrame(rows)

    def get_market_snapshot(self, codes):
        return 0, pd.DataFrame([{"code": c, "name": "", "lot_size": 1} for c in codes])

    def get_history_kl_quota(self, get_detail=False):
        details = [
            {"code": c, "name": c, "request_time": "2026-09-30 10:00:00"} for c in self.quota_codes
        ]
        return 0, (self.quota_used, self.quota_remaining, details)

    def close(self):
        self.closed = True
