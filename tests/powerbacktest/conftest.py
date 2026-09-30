from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from helpers import make_daily_bars


@pytest.fixture
def daily_bars() -> pd.DataFrame:
    return make_daily_bars()


@pytest.fixture
def fixed_today():
    return lambda tz: date(2026, 9, 30)
