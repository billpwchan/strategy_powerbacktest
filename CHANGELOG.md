# Changelog

## 1.0.0 (2026-09-30)

A full rewrite as the installable package `powerbacktest` (`pip install -e .`, command `pbt`). The 0.1 code is left in place under `src/{data,engine,strategy,utils,templates}`, `main.py`, `setup.py`, `requirements.txt` and `config.yaml` until it is deleted. The new package does not import it.

### Why 0.1 was replaced

An audit of 0.1 at commit 140aa22 found the following. Each item was reproduced by running the code against a stubbed Futu API with synthetic data.

**It did not run.** Every strategy crashed with `TypeError: BacktestReport.__init__() got an unexpected keyword argument 'fundamental_data'`, introduced in a4fdf75 (2024-12-09). LEG also crashed with `KeyError: 'close'`, and pandas 3 broke `resample("M")` and an int64 cash column.

**When patched to run, it produced wrong numbers:**

- **Timing.** Orders filled at the close of the bar that generated the signal, a same-bar look-ahead.
- **Board lots.** Every symbol used the lot size of the *last* symbol in the list; 00700 traded in multiples of 500.
- **Benchmark.** The buy-and-hold curve included the 100-day warm-up (331 bars against 260), so it was shifted about 70 bars against the strategy.
- **Report dates.** The report period showed 1970-01-01, because a RangeIndex was used as dates.
- **Warm-up.** BTSE and LEG had none, so trading began 100 days before the requested start date. MACD/MA warm-up was a fixed 100 calendar days regardless of timeframe.
- **Truncated data.** `request_history_kline` returned only its first page of 1,000 bars, about three days of 1-minute bars.
- **Metrics.**
  - Annualisation fixed at 252, so intraday Sharpe was wrong.
  - Beta was always 1.00.
  - Drawdown duration was wrong (3 bars for a 5-bar drawdown).
  - Monthly returns were labelled one month early.
  - Floating P&L was always 0.
  - Average holding time measured the gap between sells.
  - Sortino was non-standard.
- **Portfolio view.** Each symbol traded the full capital, the "portfolio" return was the average of formatted percentage strings parsed back into numbers, and portfolio drawdown was the worst single-symbol drawdown.
- **Costs.** One flat commission rate, and the configured slippage was never applied. The real HK round trip on a HK$20,000 order is about 0.37%, not 0.2%.
- **Strategies.**
  - BTSE used a causal approximation of ZIG that matched neither TDX nor a live view.
  - LEG computed MA/KDJ/MACD/Fibonacci levels but traded only a price pattern.
  - KDJ-style smoothing used a rolling mean instead of TDX `SMA`.
  - MACD emitted states while documenting crosses, and differed from futu_algo's `MACD_Cross`.
  - Parameters were defined in three places that disagreed.
- **Engineering.**
  - `setup.py` built an unimportable package.
  - Every logger call added a duplicate handler.
  - The storage and logging config sections were never read.
  - Tests covered only two indicator functions and the resampler.

### New in 1.0

- **Data.**
  - Parquet cache with coverage tracking, paging, rate limiting and quota guard.
  - Adjusted-price consistency re-download, and no caching of in-progress periods.
  - Session-aware resampling to any minute multiple.
  - Import from CSV/Parquet/TDX exports and from futu_tick_downloader tick databases.
- **Execution.**
  - Next-open fills and whole board lots, with fees fitted inside the budget.
  - Slippage in HKEX ticks by date.
  - Intrabar stop loss, trailing stop and take profit; max holding period; volume cap.
  - Fresh-signal entry rule.
- **Costs.** Dated HK statutory schedule (stamp duty, trading fee, SFC and AFRC levies, settlement fee) plus Futu broker fees; the US per-share plan with SEC fee and FINRA TAF.
- **Modes.** Scan (each symbol alone, plus a composite) and portfolio (shared cash; equal-weight, percent, fixed-value, fixed-lot or all-in sizing).
- **Strategies.**
  - TDX function library (`MA`, `EMA`, `SMA(N,M)`, `HHV`, `LLV`, `CROSS`, `BARSLAST`, `DMI`, `KDJ`, `RSI`, `MACD`, `ZIG`).
  - `point_in_time` wrapper for repainting formulas; look-ahead check on every run.
  - `macd`, `ma_cross`, `kdj`, `rsi` (futu_algo rules), `btse` (DMI + point-in-time ZIG), `leg`.
  - User strategies from files or modules.
- **Metrics.** Data-driven annualisation, peak-to-recovery drawdowns, monthly table, trade statistics with MAE/MFE, and benchmark beta, alpha and information ratio against an index or buy and hold.
- **Report.** Single-file offline HTML with equity, drawdown, monthly heatmap, metrics, a sortable symbol table, candlesticks with trades and indicators, correlation, drawdown table, assumptions and the resolved config; light and dark themes; red-up/green-up toggle. Also writes `result.json` plus CSVs.
- **Optimisation.** Grid search with an in/out-of-sample split or an anchored walk-forward, run in parallel.
- **Tooling.** `pyproject.toml` with Python 3.11+ and pandas 2.2 or 3; ruff, mypy, pytest (87 tests, including property-based accounting invariants and a regression test for every defect found in review); GitHub Actions CI.
