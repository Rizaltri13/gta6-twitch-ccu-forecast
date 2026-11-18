[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rizaltri13/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rizaltri13/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast for make.fun

Entry for the make.fun bounty **gta6-twitch-ccu-forecast-01** — forecast how many people will be watching Grand Theft Auto VI live on Twitch in each hour (2–24) of its release day, 2026-11-19 UTC, as five quantiles per hour, for the lowest pinball loss on the log scale. Prize $250; a reward needs a lower loss than both house entries (the persistence baseline and a house AI model).

## The idea

The kit's persistence baseline says every hour draws as many viewers as hour 1, with a spread widening like a random walk. That misses the shape of a release day: a rolling-midnight console launch starts the UTC day with only Oceania/Asia/Europe unlocked, dips through the quiet morning hours, then climbs steeply as North America unlocks (05:00Z/08:00Z) and peaks in the US evening.

This program fits that shape from the case's own `comparables` — 38 past big launches with full release-day hourly curves — at run time, with no network and no model call:

- For each forecast hour h = 2..24, regress the log ratio `log(v_h / v_1)` on `log(v_1)` across past launches with a real hour-1 print (≥ 5,000 viewers, so pre-unlock trickles and slow burns don't dominate the fit).
- A big hour-1 print means the launch already peaked in hour 1 (ratios stay near 1); a modest one means the big markets haven't unlocked yet (ratios rise several-fold). The per-hour slope captures exactly that.
- The fitted log ratio is shrunk 20% toward the mean ratio and clamped to the observed range (guards the extrapolation when GTA VI's hour 1 exceeds every training launch), and the spread is the per-hour residual standard deviation (floored at 0.10 log units). Quantiles: `p1 * exp(mu_h + z_tau * s_h)`.

If a case carries too few usable comparables, the program falls back to the persistence baseline, so it always writes a valid forecast.

## Validation (leave-one-launch-out on the kit's 37 usable comparables)

Mean log-pinball loss (lower is better; the house baseline's published practice loss is 0.387):

| set | persistence baseline | this model |
| --- | --- | --- |
| all 37 comparables | 0.930 | 0.645 |
| rolling-midnight console launches (8) | 0.418 | 0.301 |
| launches with hour-1 ≥ 20k viewers (27) | 0.321 | 0.250 |

It beats the baseline on every one of the 8 rolling-midnight console launches individually (e.g. ELDEN RING 0.111 vs 0.211, Cyberpunk 2077 0.139 vs 0.230, Ghost of Yōtei 0.186 vs 0.473).

## Files

- `main.py` — the forecaster (Python 3.13, standard library only).
- `LICENSE` — MIT.

## Running it

The grader runs `python3 -P -s main.py`; it reads every `<id>.in` in `/input` (or `$MAKEFUN_INPUT_DIR`) and writes `<id>.out` to `/output` (or `$MAKEFUN_OUTPUT_DIR`): 23 rows, hours 2–24, five quantiles each. The program is pure and deterministic — fixed file order, closed-form least squares, no randomness, no clock, no network — so two runs on the same case produce byte-identical output. One case runs in ~0.05 s, far inside the 8 s cap.
