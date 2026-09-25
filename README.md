# News-Based Market Regime Agent

A small daily pipeline that reads US financial news, asks an LLM (Gemini Flash) to score each headline,
and writes **one row per trading day** describing the news-implied market regime
(bearish / neutral / bullish) for the broad US market, using SPY as the proxy.

It does **not** trade and does **not** predict prices. It produces a dataset that can later
be tested against a simple price-based regime label built from SPY.

## How it works

```
Alpaca News API ──> scripts/fetch_news.py ──> data/raw/news_<date>.json        (git-ignored)
                       │
                       └──> scripts/score_news.py ──> data/scored/scored_<date>.json
                                 │
                                 └──> scripts/aggregate.py ──> data/signals.csv  (append-only)

Yahoo Finance ──> scripts/market_regime.py ──> data/market_regime.csv   (recomputed daily)
```

`run_daily.py` runs the four steps in order and stops with a non-zero exit code at the
first failure. `.github/workflows/daily.yml` runs it every weekday at 6:30 AM ET and
commits the CSVs back to the repo.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It downloads Python 3.12 and installs the pinned
dependencies from `uv.lock`.

```bash
uv sync
cp .env.example .env     # then fill in the three keys
```

`.env` keys: `ALPACA_KEY`, `ALPACA_SECRET` (any Alpaca account, paper is fine), `GEMINI_API_KEY` (Google AI Studio, free tier).

For GitHub Actions, add the same three names as repository secrets.

## Running

```bash
uv run python run_daily.py                    # today (ET)
uv run python run_daily.py --date 2026-09-24  # a specific day
uv run python scripts/fetch_news.py --date 2026-09-24 --force   # any single step
```

Every step is idempotent: if its output file already exists it exits without doing work.
Use `--force` on `fetch_news.py` / `score_news.py` to redo a day. `aggregate.py` has no
force flag on purpose (see "Signal spec").

## Signal spec (`data/signals.csv`)

One row per trading day `D`.

| column | meaning |
|---|---|
| `date` | signal date `D` (ET) |
| `regime_score` | confidence-weighted mean of headline scores, in [-1, 1] |
| `confidence` | plain mean of headline confidences, in [0, 1] |
| `label` | `bearish` if score < -0.2, `bullish` if score > 0.2, else `neutral` |
| `n_headlines` | number of headlines scored |
| `top_reasons` | the 3 headlines with the largest `confidence * abs(score)`, with their scores |
| `model`, `prompt_version` | which Claude model and prompt produced the row |
| `window_start`, `window_end` | exact news window used, ISO 8601 with offset |

**News window.** Previous weekday 16:00 ET to `D` 16:00 ET, **capped at the time the fetch
ran**. On the scheduled run this means roughly `D-1` 16:00 ET to `D` 06:30 ET: the overnight
news available before the open on `D`.

**Per-headline scoring.** Each headline + summary is sent to Gemini (`gemini-2.5-flash`, thinking off,
JSON-schema-constrained output, paced to the free-tier rate limit) with a fixed prompt that asks for `score` in [-1, 1],
`confidence` in [0, 1] and a one-sentence `reason`. The prompt tells the model to give low
confidence to single-stock, celebrity, or promotional items so they carry little weight.

**Aggregation.** `regime_score = sum(score * confidence) / sum(confidence)`. Days with no
headlines or all-zero confidence get score 0 and confidence 0.

**Append-only.** `aggregate.py` refuses to write a row for a date that already exists. Past
rows are never rewritten by the pipeline; changing history is a deliberate manual edit.

## Market regime spec (`data/market_regime.csv`)

Built from SPY adjusted daily closes (yfinance), fully recomputed each run.

| column | meaning |
|---|---|
| `ret_1d` | daily return |
| `ret_20d` | 20-trading-day return |
| `vol_20d` | annualised std of daily returns over 20 days |
| `market_label` | `bearish` if `ret_20d < -2%` **or** `vol_20d > 25%`; `bullish` if `ret_20d > +2%`; else `neutral`; blank for the first 20 warm-up rows |

The market label on day `D` uses `D`'s close. The news label on `D` is known before `D`'s
open. So comparing them on the same date is a fair test of "does pre-open news agree with
where the market is", and comparing news on `D` against market labels on `D+5` or `D+20`
tests whether news leads the regime.

## Leakage risks and how they are handled

**1. LLM lookahead bias on historical news.** The model was trained on text up to a cutoff
date. If you backfill headlines from before that cutoff, the model may "know" what happened
next, and its scores will look better than they could ever be in real time.

- The prompt instructs the model to judge only the text and not use knowledge of what
  happened afterwards. This reduces the effect but cannot eliminate it.
- Every row records `model` and `prompt_version`, so backfilled and live rows can be
  separated and the model's cutoff date compared with each row's date.
- The only fully clean evaluation set is rows produced live, day by day, after the model's
  training cutoff. Treat backfilled rows as exploratory.

**2. Timestamp leakage.** A signal dated `D` must contain only information available before
the decision point on `D`.

- The news window end is capped at the fetch time and both bounds are written into the raw
  file, the scored file and the CSV row, so it is auditable per row.
- The scheduled run happens before the market opens, so no same-session price action is in
  the headlines that feed day `D`'s row.
- Alpaca returns articles as they exist now, including later edits to the headline. For
  live-collected days this is harmless (we fetched before any edit could happen). For
  backfills it is a mild source of leakage.

**3. Price data revisions.** yfinance adjusts history for dividends and splits, so
`market_regime.csv` is recomputed from scratch rather than appended. `signals.csv` is never
recomputed.

## Repo rules

- `.env` and `data/raw/` are git-ignored. Scored files and CSVs are committed.
- `data/signals.csv` is append-only. If you change the prompt, bump `PROMPT_VERSION` in
  `scripts/score_news.py`; do not rewrite old rows.
- Scripts are small, function-based, typed, and each runs on its own with `--date`.
