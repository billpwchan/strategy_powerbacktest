# Architecture and assumptions

This document records the design decisions in the 1.0 rewrite, the assumptions every result depends on, and what is known *not* to be modelled. It is meant to be read before trusting a number from the report.

## Goals

The project is the middle of a three-repo stack: futu_tick_downloader (tick capture) → strategy_powerbacktest → futu_algo (live trading). From the original README, the old code and the other two repos, the goals are:

1. Backtest TDX-style indicator strategies on HK (and US) equities using Futu data.
2. Model what an HK retail account actually pays and can buy: board lots, stamp duty, levies, broker fees.
3. Run a strategy over many symbols and compare them in one interactive report.
4. Support intraday timeframes, including HK session-aware 2H/4H bars.
5. Store data locally instead of re-downloading it.
6. Keep strategies close enough to futu_algo's that a backtest says something about the live bot.

## Why a hand-written engine

No off-the-shelf framework fit (surveyed September 2026):

- **NautilusTrader.** Its only Futu path is a small community adapter built on the v1 API, and Nautilus v2 is dropping Python adapters.
- **QuantConnect LEAN.** No HK equities and no Futu broker.
- **backtrader and zipline.** Effectively unmaintained.
- **backtesting.py.** AGPL, and no shared cash across symbols.
- **bt.** Weight-rebalancing oriented.
- **vectorbt.** Fast, but HK stamp-duty rounding and per-order minimums need custom Numba callbacks, and its pandas ≥ 3 pin clashes with other tools.

The simulator here is about 500 lines of plain Python, covered by hand-computed tests and property tests. It is fast enough for daily bars across dozens of symbols, and for grid searches when run in parallel.

## Data flow

```
DataManager.load_bars(symbol, timeframe, start, end, warmup)
  ├─ ParquetStore.coverage(key)            what date range was already requested?
  ├─ FutuSource.fetch_bars(missing ranges) paged, rate-limited, quota-checked
  ├─ merge + adjusted-price consistency     re-download everything if old bars changed
  └─ resample_bars (non-native minutes)     session-anchored chunks
Strategy.run(bars) -> indicators, signal
check_lookahead(strategy, bars)            recompute on truncated history
Simulator(feeds).run() -> Book             fills, trades, equity, cash, exposure
book_metrics(book) -> metrics
build_payload(result) -> result.json -> report.html
```

### Cache keys and coverage

A series is identified by `(symbol, Futu K-line type, adjustment)`. Coverage is the date range that has been *requested* and fully received, which can differ from the first and last bar because holidays and pre-listing dates legitimately have none.

- **Incremental fetches** overlap the cache by 10 calendar days (21 for weekly, 62 for monthly), so the consistency check always has trading days to compare.
- **Incomplete periods** are never stored. That means bars for today, the current week and the current month; `last_complete_date` gives the cut-off.
- **Adjusted prices.** Forward-adjusted (qfq) history changes whenever a new dividend or split happens. If any overlapping close differs from the cached one, the whole series is re-downloaded. Re-requesting a symbol inside the quota window is free.

### Futu specifics used

- **Paging.** `request_history_kline(..., max_count=1000, page_req_key=...)` is paged by hand. The old code took only the first page, silently truncating intraday history to about 170 days of 60-minute bars.
- **Rate limit.** 60 requests per 30 seconds, applied per page (conservative). Rate-limit errors are retried with backoff; quota errors raise `QuotaExceededError`.
- **Quota.** `get_history_kl_quota(get_detail=True)` lists the symbols already counted in the current window. The manager allows those freely and refuses to start a new symbol at zero remaining.
- **Instrument data.** `get_stock_basicinfo(market, STOCK, code_list)` supplies name, lot size and security type, with `get_market_snapshot` as the fallback.
- **Timestamps.** `time_key` is exchange-local time (HK time for HK, US Eastern for US) and intraday bars are end-labelled. Bars are stored tz-aware.

## Timing model

At each step of the merged timeline, and for each symbol with a bar there:

1. Pending sells fill at the open, then pending buys, so freed cash is reusable.
2. Resting stop, trailing-stop and take-profit levels are checked against the bar's range.
3. The position is marked at the close, and equity is recorded.
4. The strategy's signal at this close creates the next order.

Bars are merged on their **close** time in UTC. Daily bars close at the market close (16:00 local), so HK day D is processed before US day D, and both after US day D−1.

An order decided at bar t can only fill at bar t+1 (`fill: next_open`). An order left over after a symbol's last bar expires and is logged.

**Entry blocking.** With `entry_requires_fresh_signal` (the default), a symbol is blocked at the start and after a stop, target or max-hold exit. It is unblocked on the first bar whose signal is not 1. This stops state-style strategies from buying mid-trend on day one or straight back in after a stop, while event-style strategies (crosses) are unaffected.

## Costs

`market/costs.py` holds dated schedules. The HK statutory rows are:

| From | Stamp duty | Trading fee | Tariff | SFC | AFRC | Settlement |
|:--|:--|:--|:--|:--|:--|:--|
| (start) | 0.10% | 0.005% | HK$0.50 | 0.0027% | – | 0.002%, min 2, max 100 |
| 2021-08-01 | 0.13% | | | | | |
| 2022-01-01 | | | | | 0.00015% | |
| 2023-01-01 | | 0.00565% | – | | | |
| 2023-11-17 | 0.10% | | | | | |
| 2025-06-30 | | | | | | 0.0042%, no min/max |

Rounding follows the published rules. Stamp duty rounds up to the next whole HK dollar. The other statutory fees round to the cent, with a one-cent minimum when non-zero. ETFs are stamp-duty exempt from 2015-02-13.

Broker fees default to Futu HK's fixed plan and are configurable. US uses Futu's per-share plan plus the SEC Section 31 fee and the FINRA TAF on sells (the TAF is set to zero for 2026-10-01..2026-12-31 under SR-FINRA-2026-021).

## Look-ahead detection

`check_lookahead` reruns the strategy on `bars[:t+1]` at several checkpoints and compares each indicator and signal at t with the full-history run. Any difference means a value at t depended on bars after t. The check runs on every backtest (`backtest.check_lookahead`), and a failure becomes a report warning.

Tests assert that every built-in strategy passes and that raw ZIG fails.

`tdx.point_in_time(fn, frame, window)` is the generic cure for repainting formulas. It evaluates `fn` on each bar's trailing window and keeps only the last value. The cost is O(n × window).

## Metrics

Definitions are in the module docstring of `analytics/metrics.py`. The choices that differ from common shortcuts:

- **Annualisation** uses bars per elapsed year, measured from the data. The fallback, for tests shorter than three months, is the timeframe times the market calendar.
- **Sortino** takes its downside deviation over all bars, not only losing bars.
- **Max drawdown** measures from a running peak that includes the initial capital. Its duration runs from peak to full recovery, or to the end of the test.
- **The Composite book** in scan mode is the average of the symbol books: equal capital per symbol, no rebalancing. Money figures such as fees and P&L are scaled to one book's capital.

## Known limitations

These are not modelled. Each can move results, and most are listed in the report's Assumptions panel.

- **Lot sizes are today's.** Futu does not provide lot-size history, and HKEX's board-lot reform (phase 2 from 2026-11-16) changes many lots. Override per symbol with `backtest.lot_sizes` for historical accuracy.
- **Fills use adjusted prices.** With qfq data, historical prices are scaled for later dividends, which distorts historical cash and lot arithmetic slightly (returns are unaffected). `data.adjust: none` trades raw prices but then ignores dividends.
- **Exchange rules and market structure.**
  - Suspensions: a symbol with no bar simply skips that step.
  - No price limits (none apply in HK).
  - No auction mechanics.
  - No shorting.
  - No margin.
- **Calendar gaps.** Futu's trading calendar omits ad-hoc closures such as typhoon days. They show up only as missing bars.
- **Portfolio mode currency.** It needs a single currency; mixed HK/US universes must use scan mode.
- **Not verified without a live OpenD:**
  - the exact end labels of HK 60-minute bars (assumed 10:30, 11:30, 12:00, 14:00, 15:00, 16:00);
  - how far back each K-line type goes;
  - whether Futu's native `K_120M` / `K_240M` would match the local 2H/4H resampling.

  Run `pbt data fetch` once and inspect `pbt data list` to confirm.
- **BTSE and LEG** are reconstructions: the original TDX sources are not in the repository.
- **The US fee cap** (`max_pct_of_value`) is from Futu's published schedule and applied to commission and platform fee separately. Confirm it against your own statements.
