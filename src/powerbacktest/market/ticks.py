"""Minimum price increments, used to express slippage in ticks.

HK uses the HKEX spread table for equities, which HKEX narrowed in two steps
(2025-08-04 for the HK$10-50 bands, 2026-08-03 for the HK$0.50-10 band). The date of the
bar decides which table applies. US equities trade in cents above $1 and 1/100 cent below.
"""

from __future__ import annotations

import bisect
from datetime import date

from powerbacktest.market.instrument import Instrument

# (upper price bound inclusive, tick size) bands; the first band whose bound >= price applies.
_HK_BASE: tuple[tuple[float, float], ...] = (
    (0.25, 0.001),
    (0.50, 0.005),
    (10.00, 0.010),
    (20.00, 0.020),
    (100.00, 0.050),
    (200.00, 0.100),
    (500.00, 0.200),
    (1000.00, 0.500),
    (2000.00, 1.000),
    (5000.00, 2.000),
    (float("inf"), 5.000),
)
_HK_2025: tuple[tuple[float, float], ...] = (
    (0.25, 0.001),
    (0.50, 0.005),
    (10.00, 0.010),
    (20.00, 0.010),
    (50.00, 0.020),
    (100.00, 0.050),
    (200.00, 0.100),
    (500.00, 0.200),
    (1000.00, 0.500),
    (2000.00, 1.000),
    (5000.00, 2.000),
    (float("inf"), 5.000),
)
_HK_2026: tuple[tuple[float, float], ...] = (
    (0.25, 0.001),
    (0.50, 0.005),
    (10.00, 0.005),
    *_HK_2025[3:],
)
HK_SPREAD_TABLES: tuple[tuple[date, tuple[tuple[float, float], ...]], ...] = (
    (date(1990, 1, 1), _HK_BASE),
    (date(2025, 8, 4), _HK_2025),
    (date(2026, 8, 3), _HK_2026),
)


def hk_tick_size(price: float, when: date) -> float:
    idx = bisect.bisect_right([d for d, _ in HK_SPREAD_TABLES], when) - 1
    table = HK_SPREAD_TABLES[max(idx, 0)][1]
    for bound, tick in table:
        if price <= bound + 1e-12:
            return tick
    return table[-1][1]


def us_tick_size(price: float) -> float:
    return 0.01 if price >= 1.0 else 0.0001


def tick_size(price: float, instrument: Instrument, when: date) -> float:
    if instrument.market.code == "HK":
        return hk_tick_size(price, when)
    return us_tick_size(price)
