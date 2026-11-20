// GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).
//
// Forecasts, for every PS5 trophy of Grand Theft Auto VI, its Exophase earn
// rate at the case's target, as five quantiles (5/25/50/75/95%) of the rate
// in percent. Scored by mean pinball loss on the logit scale.
//
// The model, per trophy, in logit space:
//   - a Theil-Sen robust line of logit(rate) against ln(1 + hours since
//     2026-11-19T00:00Z), fitted on the case's history; the median at the
//     target is the line read at the target. Theil-Sen resists the discrete
//     drops Exophase's refreshes cause, which corrupt a least-squares fit.
//   - the spread combines the baseline's random-walk term (0.5*sqrt(days)),
//     the fit's residual scale, slope uncertainty times the extrapolation
//     distance, and -- where the history shows refresh jumps -- the variance
//     of the jumps still to come before the target.
//   - when the history is flat (no refresh yet, every rate unchanged, short
//     horizon -- the launch-night practice shape), the answer is a mixture:
//     no refresh before the target (rates stay) or a refresh (newly tracked
//     players dilute every rate to about a fifth). The quantiles invert that
//     mixture, so the band covers both outcomes instead of betting on one.
//
// Node 22, standard library only, no network, deterministic: two runs of
// the same case write byte-identical output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release-day origin for the log-time axis (the house curve baseline's). */
const ORIGIN_MS = Date.parse('2026-11-19T00:00:00Z');

interface Trophy { id: number; name: string; tier: string }
interface Read { slot: string; at: string; players_tracked: number; rates: Array<number | null> }
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

const median = (v: number[]): number => {
  if (v.length === 0) return 0;
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
};

/** Standard normal CDF (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7). */
const normCdf = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
};

/**
 * Theil-Sen slope of ys on xs (median of pairwise slopes). Subsamples to
 * at most 80 points so the pair count stays small on long histories.
 */
const theilSen = (xs: number[], ys: number[]): number => {
  const n = xs.length;
  if (n < 2) return 0;
  let px = xs;
  let py = ys;
  if (n > 80) {
    const idx: number[] = [];
    for (let i = 0; i < 80; i++) idx.push(Math.round((i * (n - 1)) / 79));
    px = idx.map((i) => xs[i]);
    py = idx.map((i) => ys[i]);
  }
  const slopes: number[] = [];
  for (let i = 0; i < px.length; i++) {
    for (let j = i + 1; j < px.length; j++) {
      const dx = px[j] - px[i];
      if (Math.abs(dx) > 1e-9) slopes.push((py[j] - py[i]) / dx);
    }
  }
  return slopes.length === 0 ? 0 : median(slopes);
};

/**
 * One quantile of a two-component normal mixture on the logit scale:
 * with probability (1 - p) the answer sits near m0, with probability p
 * near m1 (a refresh), each with spread s0. Inverts the mixture CDF by
 * bisection.
 */
const mixtureQuantile = (m0: number, m1: number, p: number, s0: number, tau: number): number => {
  let lo = Math.min(m0, m1) - 8 * s0 - 2;
  let hi = Math.max(m0, m1) + 8 * s0 + 2;
  for (let k = 0; k < 90; k++) {
    const mid = 0.5 * (lo + hi);
    const cdf = (1 - p) * normCdf((mid - m0) / s0) + p * normCdf((mid - m1) / s0);
    if (cdf < tau) lo = mid;
    else hi = mid;
  }
  return invLogit(0.5 * (lo + hi));
};

/** Hours from the release-day origin to an ISO timestamp. */
const hoursSinceOrigin = (iso: string): number =>
  (Date.parse(iso) - ORIGIN_MS) / 3_600_000;
/** Log-time axis: ln(1 + hours since the origin). */
const logTime = (iso: string): number => Math.log(Math.max(0.05, 1 + hoursSinceOrigin(iso)));

/**
 * Forecast one case. Returns the { trophies: [{id, q: [q05..q95]}] } object
 * the scorer reads.
 */
const forecastCase = (c: Case): { trophies: Array<{ id: number; q: number[] }> } => {
  const hist = c.history;
  const horizonH = c.horizon_minutes / 60;
  const xTarget = logTime(c.target);

  // Refresh jumps in the history: consecutive reads where the tracked-player
  // count jumps by more than 3%. Each event records the median logit move
  // across the trophies (dilution drops rates; the median resists trophies
  // that moved for other reasons).
  const jumpSizes: number[] = [];
  for (let i = 1; i < hist.length; i++) {
    const n0 = hist[i - 1].players_tracked;
    const n1 = hist[i].players_tracked;
    if (!n0 || (n1 - n0) / n0 <= 0.03) continue;
    const ds: number[] = [];
    for (let k = 0; k < c.trophies.length; k++) {
      const r0 = hist[i - 1].rates[k];
      const r1 = hist[i].rates[k];
      if (r0 !== null && r1 !== null && r0 < 99.99) ds.push(logit(r1) - logit(r0));
    }
    if (ds.length > 0) jumpSizes.push(median(ds));
  }
  const spanH =
    hist.length > 1
      ? (Date.parse(hist[hist.length - 1].at) - Date.parse(hist[0].at)) / 3_600_000
      : 0;
  const jumpRate = spanH > 0 ? jumpSizes.length / spanH : 0;
  let jumpSd = 0;
  if (jumpSizes.length >= 2) {
    const mu = jumpSizes.reduce((a, b) => a + b, 0) / jumpSizes.length;
    jumpSd = Math.sqrt(
      jumpSizes.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (jumpSizes.length - 1),
    );
  }

  const out: Array<{ id: number; q: number[] }> = [];
  for (let ti = 0; ti < c.trophies.length; ti++) {
    const t = c.trophies[ti];
    // Most recent non-null rate for this trophy.
    let lastRate: number | null = null;
    for (let i = hist.length - 1; i >= 0; i--) {
      if (hist[i].rates[ti] !== null) {
        lastRate = hist[i].rates[ti] as number;
        break;
      }
    }
    // A trophy nobody has earned yet sits at (or near) 100%: forecast it
    // tight at the top; a refresh cannot push it down.
    if (lastRate !== null && lastRate >= 99.99) {
      out.push({ id: t.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] });
      continue;
    }

    const series: Array<[number, number]> = [];
    for (const h of hist) {
      const r = h.rates[ti];
      if (r === null) continue;
      series.push([logTime(h.at), logit(r)]);
    }
    if (series.length === 0) {
      // No valid read for this trophy at all: neutral prior, wide spread.
      const q = Z.map((z) => round4(invLogit(logit(1) + z * 1.5)));
      out.push({ id: t.id, q: q.sort((a, b) => a - b).map((v) => Math.max(0, Math.min(100, v))) });
      continue;
    }

    const xs = series.map((s) => s[0]);
    const ys = series.map((s) => s[1]);
    const b = series.length >= 3 ? theilSen(xs, ys) : 0;
    const a = median(ys.map((y, i) => y - b * xs[i]));
    const m = a + b * xTarget;

    // Spread: random-walk term + residual scale + slope uncertainty over
    // the extrapolation distance, in quadrature.
    const resid = ys.map((y, i) => y - (a + b * xs[i]));
    const residMed = median(resid);
    const sigResid = Math.min(
      1.4826 * median(resid.map((r) => Math.abs(r - residMed))),
      1,
    );
    const xBar = xs.reduce((s, v) => s + v, 0) / xs.length;
    const xSd = Math.sqrt(xs.reduce((s, v) => s + (v - xBar) * (v - xBar), 0) / xs.length);
    const sigSlope = xSd > 0 && xs.length > 2 ? sigResid / (xSd * Math.sqrt(xs.length)) : 0;
    const sigExtrap = Math.abs(sigSlope * (xTarget - xBar));
    const sigRw = 0.5 * Math.sqrt(c.horizon_minutes / 1440);
    let sig0 = Math.sqrt(sigRw * sigRw + sigResid * sigResid + sigExtrap * sigExtrap);
    sig0 = Math.min(Math.max(sig0, 0.15), 2);

    // Flat history, no refresh seen yet, short horizon: the launch-night
    // shape. Exophase refreshes hours apart and the tracked count has been
    // frozen the whole history, so a refresh before the target is a live
    // possibility -- and a refresh dilutes every rate: newly tracked players
    // hold far fewer trophies, so the refreshed rate is about a fifth of the
    // stale one (a batch ~5x the tracked count, new players at ~7% of the
    // incumbents' earn rates). Forecast the mixture, not one branch.
    const yRange = Math.max(...ys) - Math.min(...ys);
    if (jumpSizes.length === 0 && yRange < 0.05 && horizonH <= 12 && lastRate !== null) {
      const f = (1 + 5 * 0.07) / 6; // ~0.225 dilution factor
      const m0 = logit(lastRate);
      const m1 = logit(lastRate * f);
      const q = LEVELS.map((tau) => round4(mixtureQuantile(m0, m1, 0.5, sig0, tau)));
      out.push({ id: t.id, q: q.sort((x, y) => x - y).map((v) => Math.max(0, Math.min(100, v))) });
      continue;
    }

    // Data branch: the trend is the median; refreshes still to come add
    // variance around it (their mean is already in the trend).
    if (jumpSizes.length > 0) {
      const kExp = jumpRate * horizonH;
      sig0 = Math.min(Math.sqrt(sig0 * sig0 + kExp * jumpSd * jumpSd), 2.5);
    }
    const q = Z.map((z) => round4(invLogit(m + z * sig0)));
    out.push({ id: t.id, q: q.sort((x, y) => x - y).map((v) => Math.max(0, Math.min(100, v))) });
  }
  return { trophies: out };
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(forecastCase(c))}\n`);
}
