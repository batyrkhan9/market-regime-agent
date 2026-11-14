# Stability test: 2026-10-26, 5 runs

- Model `claude-sonnet-5-5`, prompt `v2-batched`, batch size 25, 1000 headlines in the saved file's order.
- Input: headlines only from the saved scored file (summaries were not stored), so every item was sent with `Summary: (none)`. Not comparable to the production scores for this date.
- Input SHA-256 (rendered batch text): `1d8693dbbda55fea41a51d519f0a29099848aa3daca12d730dcb5168b1be5bd7`
- Daily score = confidence-weighted mean score (as in aggregate.py). Labels: bearish < -0.2, bullish > 0.2. Standard deviations are sample (n-1).

## 1. Daily score per run

| Run | Score (all) | Label | Score (conf >= 0.3) | n | Label |
|---|---|---|---|---|---|
| 1 | -0.0261 | neutral | -0.2524 | 17 | bearish |
| 2 | -0.0263 | neutral | -0.2606 | 15 | bearish |
| 3 | -0.0257 | neutral | -0.2535 | 16 | bearish |
| 4 | -0.0231 | neutral | -0.2233 | 17 | bearish |
| 5 | -0.0234 | neutral | -0.2443 | 16 | bearish |

## 2. Across runs

| Daily score | Min | Max | Range | Std | Label changed? |
|---|---|---|---|---|---|
| All headlines | -0.0263 | -0.0231 | 0.0032 | 0.0015 | no (neutral) |
| Conf >= 0.3 | -0.2606 | -0.2233 | 0.0373 | 0.0144 | no (bearish) |

## 3. Per headline

| Metric | Value |
|---|---|
| Mean std of score across runs | 0.0042 |
| Mean std of confidence across runs | 0.0077 |
| Headlines with identical score in all runs | 69.3% |
| Headlines with identical confidence in all runs | 24.3% |

## 4. Label flips

0.6% of headlines (6 of 1000) changed bearish/neutral/bullish label in at least one run.

Top 5 headlines by score range across runs:

| Headline | Scores (runs 1-5) | Range | Labels |
|---|---|---|---|
| Chevron, Exxon, Shell and ConocoPhillips Hit Record Highs as Crude Rally Powers On Into Earnings | -0.10, -0.10, +0.10, -0.10, -0.15 | 0.25 | neutral, neutral, neutral, neutral, neutral |
| Pentagon Prepares New Iran Strike Options for Trump as Oil Prices Surge: Report | -0.40, -0.40, -0.50, -0.50, -0.35 | 0.15 | bearish, bearish, bearish, bearish, bearish |
| A $1 Trillion Depreciation Bomb Is Ticking Beneath the AI Market | -0.30, -0.15, -0.25, -0.20, -0.30 | 0.15 | bearish, neutral, bearish, neutral, bearish |
| Fed, ECB Expected To Hold Rates Next Week, But December Hikes Still In Play | -0.10, -0.20, -0.10, -0.10, -0.10 | 0.10 | neutral, neutral, neutral, neutral, neutral |
| Trump Eases 50% Tariffs on Select Canadian Goods Days Before Midterms, While Susan Collins Faces Tough Reelection Fight | +0.20, +0.20, +0.15, +0.25, +0.25 | 0.10 | neutral, neutral, neutral, bullish, bullish |

## 5. Cost

Total $2.8191 for 200 requests: 254,560 input, 228,091 output, 1,302 cache-write, 128,898 cache-read tokens. Prices (USD/MTok): input 2.0, output 10.0, cache_write_5m 2.5, cache_read 0.2.
Batch retries: 0.

## Notes

- The conf >= 0.3 score rests on very few headlines. 14 headlines cleared the cutoff in every run and 6 crossed it in some runs only, so part of that score's run-to-run spread comes from the subset changing.
