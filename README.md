# gta6-trophy-rarity-forecast-agent

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SIRIBRAHEEM/gta6-trophy-rarity-forecast-agent/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SIRIBRAHEEM/gta6-trophy-rarity-forecast-agent/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's Exophase earn rate one week after launch, from the house's recording of the trophy page. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` (Node 22, standard library only) reads every `<id>.in` case in the input directory and writes `<id>.out` — `{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}`, one row per trophy. Offline, deterministic: two runs of the same case write byte-identical output. Runs in ~0.2s per case (cap 8s).

## The model

An earn rate is a ratio, and its two parts move differently: the tracked-player pool grows in jumps whenever Exophase refreshes its counts (diluting every rate), while earned counts only ever grow as players play. So the forecaster splits the rate into its two parts and models each separately — the earned count of every trophy and the tracked-player count, each fitted as a straight line of its log against log(1 + hours since release), with earned-count slopes shrunk toward their tier's mean so rare trophies borrow strength from their tier. The median rate is the two forecasts divided. The spread is honest about three things: a random walk widening with the horizon, the fitted slopes' uncertainty over the extrapolation, and the refresh noise in the case's own history. When the horizon is shorter than one refresh cycle and the page has gone stale, a refresh is overdue and the forecast becomes a mixture — no refresh before the target, or a refresh diluting every rate by the usual drop — and the quantiles invert that mixture instead of betting on one branch. Practice score: pooled pinball loss 0.23 vs the house baselines' 1.51 (0.16–0.30 across reconstruction variants; the kit holds no local scorer, so the practice loss is estimated from the refresh the rules publish — 638 → 3,868 tracked players, every rate falling).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
