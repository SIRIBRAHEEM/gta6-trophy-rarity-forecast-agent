// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
// rate on Exophase at the case's target, as five quantiles (5/25/50/75/95%) of
// the rate in percent. Score is the mean pinball loss on the logit scale.
//
// How it works, in one paragraph. An earn rate is a ratio: the players who
// earned a trophy over the players Exophase tracks. The two move differently —
// the tracked pool grows in jumps whenever Exophase refreshes its counts
// (diluting every rate), while the earned counts only ever grow as players
// play. So this forecaster splits the rate into its two parts and models each
// separately: the earned count of every trophy and the tracked-player count,
// each fitted as a straight line of its log against log(1 + hours since
// release), the earned-count slopes shrunk toward their tier's mean slope so
// rare trophies with little data borrow strength from their tier. The median
// rate is the two forecasts divided. The spread around it is honest about
// three things: a random walk that widens with the horizon, the uncertainty of
// the fitted slopes over the extrapolation, and how much the counts jitter
// around their trends (the refresh noise in the case's own history). On top of
// that, when the horizon is shorter than one refresh cycle and the page has
// gone stale — no refresh for much longer than the usual gap between
// refreshes — a refresh is overdue, and the forecast becomes a mixture: with
// some probability no refresh comes before the target and the trend holds,
// with the rest a refresh lands and dilutes every rate by the drop a refresh
// usually brings. The five quantiles are the quantiles of that mixture, so
// the band widens and skews down instead of pretending the rate will sit
// still. With fresh data, or over a long horizon where the trend already
// absorbs the refreshes, the mixture switches off and the trend does the work.
//
// Usage: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, in logit units per square-root day (the baseline's). */
const SIGMA_RW = 0.5;
/** Prior per-refresh logit jump when the history holds no refresh. */
const JUMP_PRIOR = 0.8;
/** Prior gap between refreshes, in hours, when the history holds none. */
const INTERVAL_PRIOR_H = 4;
/** Prior logit drop one refresh brings, when the history holds no refresh. */
const DILUTION_PRIOR = 1.5;
/** Prior strength (in distinct values) when shrinking a slope to its tier mean. */
const SHRINK_K = 2;
/** The tracked-player count never grows by more than this factor over the horizon. */
const N_GROWTH_CAP = 4;
/** An earned count never exceeds this share of the tracked players. */
const E_CAP_FRAC = 0.995;
/** The spread never goes below this, in logit units. */
const SPREAD_MIN = 0.2;
/** The spread never goes above this, in logit units. */
const SPREAD_MAX = 2.5;

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Median of a list of numbers. */
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
/** Mean of a list of numbers. */
const mean = (xs: number[]): number => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
/** The standard normal's cumulative distribution, by a rational approximation. */
const normCdf = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
};

/** Least-squares line of y on x. Returns slope, intercept and residual spread. */
const fitLine = (xs: number[], ys: number[]): { slope: number; intercept: number; resid: number; slopeSe: number } => {
  const n = xs.length;
  if (n < 2) return { slope: 0, intercept: n === 1 ? ys[0] : 0, resid: 0, slopeSe: 0 };
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let sse = 0;
  for (let i = 0; i < n; i++) sse += (ys[i] - (intercept + slope * xs[i])) ** 2;
  const resid = Math.sqrt(sse / Math.max(1, n - 2));
  return { slope, intercept, resid, slopeSe: sxx > 0 ? resid / Math.sqrt(sxx) : 0 };
};

/** One distinct page: the reads collapse to the moments the page actually changed. */
interface Snap {
  /** When this page was first seen (for staleness: the page has held since then). */
  firstMs: number;
  /** When this page was last seen (for fitting: the value held until then). */
  atMs: number;
  x: number;
  players: number;
  rates: number[];
}

const forecastCase = (c: Case): Array<{ id: number; q: number[] }> => {
  const releaseMs = Date.parse(c.game.release_date);
  const originMs = Number.isFinite(releaseMs) ? releaseMs : Date.parse('2026-11-19T00:00:00Z');
  const atMs = Date.parse(c.at);
  const horizonDays = c.horizon_minutes / 1440;
  const horizonHours = c.horizon_minutes / 60;
  const xTarget = Math.log(1 + Math.max(0, (atMs + c.horizon_minutes * 60000 - originMs) / 3600000));

  // Collapse the history to distinct pages, keeping the last time each page was seen.
  const snaps: Snap[] = [];
  for (const h of c.history) {
    const rates = h.rates.map((r) => (r === null ? 0 : r));
    const prev = snaps[snaps.length - 1];
    if (prev !== undefined && prev.players === h.players_tracked && prev.rates.every((r, i) => r === rates[i])) {
      prev.atMs = Date.parse(h.at);
      prev.x = Math.log(1 + Math.max(0, (prev.atMs - originMs) / 3600000));
      continue;
    }
    const at = Date.parse(h.at);
    snaps.push({ firstMs: at, atMs: at, x: Math.log(1 + Math.max(0, (at - originMs) / 3600000)), players: h.players_tracked, rates });
  }
  if (snaps.length === 0) throw new Error(`${c.id} has no history`);
  const last = snaps[snaps.length - 1];
  const lastX = last.x;
  const lastN = Math.max(1, last.players);

  // Refreshes: the moments the page changed, how far apart, and what each did to the rates.
  const changeTimes: number[] = [];
  const refreshDrops: number[] = [];
  for (let s = 1; s < snaps.length; s++) {
    const prev = snaps[s - 1];
    const cur = snaps[s];
    changeTimes.push(cur.firstMs);
    const drops: number[] = [];
    for (let i = 0; i < c.trophies.length; i++) {
      if (prev.rates[i] <= 0 || cur.rates[i] <= 0 || prev.rates[i] >= 100 || cur.rates[i] >= 100) continue;
      drops.push(-(logit(cur.rates[i]) - logit(prev.rates[i])));
    }
    if (drops.length > 0) refreshDrops.push(median(drops));
  }
  const gaps: number[] = [];
  for (let s = 1; s < changeTimes.length; s++) gaps.push((changeTimes[s] - changeTimes[s - 1]) / 3600000);
  const intervalH = gaps.length > 0 ? median(gaps) : INTERVAL_PRIOR_H;
  const dropTypical = refreshDrops.length > 0 ? Math.max(0, median(refreshDrops)) : DILUTION_PRIOR;

  // The tracked-player count: log-linear in log-time, anchored at the last read.
  const xs = snaps.map((s) => s.x);
  const nFit = fitLine(xs, snaps.map((s) => Math.log(Math.max(1, s.players))));
  const nDistinct = new Set(snaps.map((s) => s.players)).size;
  const nSlope = nFit.slope * (nDistinct / (nDistinct + 1));
  const logNTarget = Math.log(lastN) + nSlope * (xTarget - lastX);
  const nTarget = Math.min(lastN * N_GROWTH_CAP, Math.max(lastN, Math.exp(logNTarget)));

  // Each trophy's earned count: log-linear in log-time, slope shrunk to its tier mean.
  const slopes: number[] = [];
  const slopeSes: number[] = [];
  const eResids: number[] = [];
  const nDistinctPerTrophy: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const ys: number[] = [];
    const xsi: number[] = [];
    for (const s of snaps) {
      const e = Math.max(0.5, (s.rates[i] / 100) * s.players);
      ys.push(Math.log(e));
      xsi.push(s.x);
    }
    nDistinctPerTrophy.push(new Set(ys.map((y) => Math.round(y * 1000))).size);
    const fit = fitLine(xsi, ys);
    slopes.push(fit.slope);
    slopeSes.push(fit.slopeSe);
    eResids.push(fit.resid);
  }
  const tierMean: Record<string, number | null> = {};
  for (const tier of [...new Set(c.trophies.map((t) => t.tier))]) {
    const vals: number[] = [];
    c.trophies.forEach((t, i) => {
      if (t.tier === tier && nDistinctPerTrophy[i] >= 2) vals.push(slopes[i]);
    });
    tierMean[tier] = vals.length > 0 ? mean(vals) : null;
  }
  const globalVals = slopes.filter((_, i) => nDistinctPerTrophy[i] >= 2);
  const globalMean = globalVals.length > 0 ? mean(globalVals) : 0;

  // How stale is the page, and is a refresh overdue before the target? The page
  // has held since the last snap was FIRST seen, not since it was last seen.
  const lastChangeMs = changeTimes.length > 0 ? changeTimes[changeTimes.length - 1] : snaps[0].firstMs;
  const staleHours = Math.max(0, (atMs - lastChangeMs) / 3600000);
  const timeToNext = Math.max(0.5, intervalH - staleHours);
  const pRefresh = horizonHours < intervalH ? 1 - Math.exp(-horizonHours / timeToNext) : 0;

  // The spread: random walk + slope uncertainty + the counts' jitter around their trends.
  const sRw = SIGMA_RW * Math.sqrt(horizonDays);
  const sResid = Math.sqrt(median(eResids) ** 2 + nFit.resid ** 2);

  return c.trophies.map((t, i) => {
    const tier = tierMean[t.tier] ?? globalMean;
    const w = nDistinctPerTrophy[i] / (nDistinctPerTrophy[i] + SHRINK_K);
    const slope = w * slopes[i] + (1 - w) * tier;
    const lastE = Math.max(0.5, (last.rates[i] / 100) * lastN);
    let eTarget = Math.exp(Math.log(lastE) + slope * (xTarget - lastX));
    eTarget = Math.min(E_CAP_FRAC * nTarget, Math.max(0.5, eTarget));
    const medianRate = clamp((eTarget / nTarget) * 100);
    const sSlope = Math.min(1.5, slopeSes[i] * (xTarget - lastX));
    const s0 = Math.min(SPREAD_MAX, Math.max(SPREAD_MIN, Math.sqrt(sRw * sRw + sSlope * sSlope + sResid * sResid)));
    const l0 = logit(medianRate);
    let q: number[];
    if (pRefresh > 0.01) {
      // Mixture: with probability 1-p the trend holds, with probability p a
      // refresh lands first and dilutes the rate by dropTypical logit units.
      const lo = l0 - dropTypical - 5 * s0;
      const hi = l0 + 5 * s0;
      const cdf = (x: number): number => (1 - pRefresh) * normCdf((x - l0) / s0) + pRefresh * normCdf((x - (l0 - dropTypical)) / s0);
      q = LEVELS.map((tau) => {
        let a = lo;
        let b = hi;
        for (let k = 0; k < 80; k++) {
          const m = (a + b) / 2;
          if (cdf(m) < tau) a = m;
          else b = m;
        }
        return round4(clamp(invLogit((a + b) / 2)));
      });
    } else {
      q = Z.map((z) => round4(clamp(invLogit(l0 + z * s0))));
    }
    // Keep the row non-decreasing after rounding.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  if (trophies.length !== c.trophies.length) throw new Error(`${file} forecast ${trophies.length} trophies, case has ${c.trophies.length}`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
