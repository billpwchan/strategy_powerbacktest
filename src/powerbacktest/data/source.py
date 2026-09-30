"""Data sources: the protocol and the Futu OpenD implementation.

``futu`` is imported lazily so that everything except live fetching works without OpenD,
and so that tests can inject a fake quote context.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol

import pandas as pd

from powerbacktest.data.schema import to_bars
from powerbacktest.data.store import InstrumentInfo
from powerbacktest.errors import DataSourceError, QuotaExceededError
from powerbacktest.market.instrument import MARKETS, parse_symbol

log = logging.getLogger(__name__)

# Futu AuType values are plain strings: AuType.QFQ == "qfq", AuType.NONE == "None".
AUTYPE = {"qfq": "qfq", "hfq": "hfq", "none": "None"}
RET_OK = 0
PAGE_SIZE = 1000
_RATE_LIMIT_MARKERS = ("频率", "頻率", "frequen", "too many", "限频", "rate limit")
_QUOTA_MARKERS = ("额度", "額度", "quota")


@dataclass
class QuotaStatus:
    used: int
    remaining: int
    codes: set[str] = field(default_factory=set)


class DataSource(Protocol):
    name: str

    def fetch_bars(
        self, symbol: str, ktype: str, adjust: str, start: date, end: date
    ) -> pd.DataFrame: ...

    def fetch_instruments(self, symbols: list[str]) -> dict[str, InstrumentInfo]: ...

    def quota(self) -> QuotaStatus | None: ...

    def close(self) -> None: ...


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` in any ``window`` seconds."""

    def __init__(
        self,
        max_calls: int,
        window: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_calls = max_calls
        self.window = window
        self.clock = clock
        self.sleep = sleep
        self.calls: deque[float] = deque()

    def acquire(self) -> None:
        now = self.clock()
        while self.calls and now - self.calls[0] >= self.window:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            wait = self.window - (now - self.calls[0]) + 0.05
            log.info("Futu rate limit reached; waiting %.1fs", wait)
            self.sleep(wait)
            now = self.clock()
            while self.calls and now - self.calls[0] >= self.window:
                self.calls.popleft()
        self.calls.append(now)


def _default_context_factory(host: str, port: int) -> Any:
    try:
        from futu import OpenQuoteContext
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataSourceError("futu-api is not installed; pip install futu-api") from exc
    return OpenQuoteContext(host=host, port=port)


class FutuSource:
    name = "futu"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        *,
        rate_limit: int = 60,
        rate_window: float = 30.0,
        max_retries: int = 3,
        context_factory: Callable[[str, int], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self._factory = context_factory or _default_context_factory
        self._ctx: Any = None
        self._sleep = sleep
        self.limiter = RateLimiter(rate_limit, rate_window, sleep=sleep)
        self.requests = 0

    # ------------------------------------------------------------ lifecycle

    def _context(self) -> Any:
        if self._ctx is None:
            log.info("Connecting to Futu OpenD at %s:%s", self.host, self.port)
            try:
                self._ctx = self._factory(self.host, self.port)
            except Exception as exc:
                raise DataSourceError(
                    f"Cannot connect to Futu OpenD at {self.host}:{self.port}: {exc}. "
                    "Start OpenD and log in, or run with --offline to use cached data."
                ) from exc
        return self._ctx

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.close()
            finally:
                self._ctx = None

    def __enter__(self) -> FutuSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --------------------------------------------------------------- calls

    def _call(self, what: str, fn: Callable[[], tuple[Any, ...]]) -> tuple[Any, ...]:
        attempt = 0
        while True:
            self.limiter.acquire()
            self.requests += 1
            result = fn()
            ret, data = result[0], result[1]
            if ret == RET_OK:
                return result
            message = str(data)
            lowered = message.lower()
            if any(m in lowered for m in _QUOTA_MARKERS):
                raise QuotaExceededError(f"{what}: {message}")
            if any(m in lowered for m in _RATE_LIMIT_MARKERS) and attempt < self.max_retries:
                attempt += 1
                wait = self.limiter.window * attempt
                log.warning(
                    "%s rate-limited by OpenD (%s); retry %d in %.0fs", what, message, attempt, wait
                )
                self._sleep(wait)
                continue
            raise DataSourceError(f"{what} failed: {message}")

    def fetch_bars(
        self, symbol: str, ktype: str, adjust: str, start: date, end: date
    ) -> pd.DataFrame:
        market = MARKETS[parse_symbol(symbol)[0]]
        ctx = self._context()
        frames: list[pd.DataFrame] = []
        page_key: Any = None
        pages = 0
        while True:
            key = page_key
            _, data, page_key = self._call(
                f"request_history_kline({symbol}, {ktype}, {adjust}, {start}..{end})",
                lambda key=key: ctx.request_history_kline(  # type: ignore[misc]
                    symbol,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    ktype=ktype,
                    autype=AUTYPE[adjust],
                    max_count=PAGE_SIZE,
                    page_req_key=key,
                ),
            )
            pages += 1
            if data is not None and len(data):
                frames.append(data)
            if page_key is None:
                break
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        bars = to_bars(raw, market.tz)
        log.info(
            "Fetched %s %s %s %s..%s: %d bars in %d page(s)",
            symbol,
            ktype,
            adjust,
            start,
            end,
            len(bars),
            pages,
        )
        return bars

    def fetch_instruments(self, symbols: list[str]) -> dict[str, InstrumentInfo]:
        if not symbols:
            return {}
        ctx = self._context()
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        out: dict[str, InstrumentInfo] = {}
        by_market: dict[str, list[str]] = {}
        for sym in symbols:
            by_market.setdefault(parse_symbol(sym)[0], []).append(sym)
        for market, codes in by_market.items():
            try:
                _, frame = self._call(
                    f"get_stock_basicinfo({market})",
                    lambda market=market, codes=codes: ctx.get_stock_basicinfo(  # type: ignore[misc]
                        market, "STOCK", codes
                    ),
                )
            except DataSourceError as exc:
                log.warning("get_stock_basicinfo failed for %s: %s", codes, exc)
                continue
            for row in frame.to_dict("records"):
                sym = str(row["code"])
                out[sym] = InstrumentInfo(
                    symbol=sym,
                    name=str(row.get("name") or ""),
                    lot_size=max(int(row.get("lot_size") or 1), 1),
                    security_type=str(row.get("stock_type") or "STOCK"),
                    listing_date=str(row.get("listing_date") or "") or None,
                    updated_at=stamp,
                )
        missing = [s for s in symbols if s not in out]
        if missing:
            try:
                _, snap = self._call(
                    "get_market_snapshot", lambda: ctx.get_market_snapshot(missing)
                )
                for row in snap.to_dict("records"):
                    sym = str(row["code"])
                    out[sym] = InstrumentInfo(
                        symbol=sym,
                        name=str(row.get("name") or ""),
                        lot_size=max(int(row.get("lot_size") or 1), 1),
                        security_type="STOCK" if row.get("equity_valid", True) else "OTHER",
                        listing_date=None,
                        updated_at=stamp,
                    )
            except DataSourceError as exc:
                log.warning("get_market_snapshot failed for %s: %s", missing, exc)
        return out

    def quota(self) -> QuotaStatus | None:
        ctx = self._context()
        try:
            _, (used, remaining, details) = self._call(
                "get_history_kl_quota", lambda: ctx.get_history_kl_quota(get_detail=True)
            )
        except DataSourceError as exc:
            log.warning("Could not read historical K-line quota: %s", exc)
            return None
        return QuotaStatus(
            used=int(used),
            remaining=int(remaining),
            codes={str(d.get("code")) for d in details or []},
        )
