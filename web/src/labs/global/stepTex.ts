/**
 * "This step": the update rule of the focused method with the numbers of the current step filled
 * in (KaTeX source + one plain sentence), read from Step.info.
 */
import type { Step, Vector } from '../../core/types';
import { int, sci, sig, superscript, vec } from '../../core/format';
import { cmaConstants } from '../../methods/unconstrained/global_';
import { acceptedCount, deHighlight, searchScale } from './geometry';

/**
 * A computed number in TeX: `digits` significant digits, ×10ⁿ below 10⁻³ and at or above 10⁵.
 * Trailing zeros are kept (1.000, not 1), so a rounded value never looks exact.
 */
export function tn(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(digits - 1).split('e');
    return `${m}\\times 10^{${Number(e)}}`;
  }
  const r = v.toPrecision(digits);
  // toPrecision can switch to exponent form near 10⁵ after rounding (99999.9 → "1.000e+5").
  return r.includes('e') ? tn(Number(r), digits) : r;
}

/** A parameter value as the user set it (shortest exact form, at most 6 significant digits). */
export function tp(v: number): string {
  if (!Number.isFinite(v)) return tn(v);
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [m, e] = v.toExponential(5).split('e');
    const mant = m.replace(/\.?0+$/, '');
    return `${mant === '1' ? '' : mant === '-1' ? '-' : `${mant}\\times `}10^{${Number(e)}}`;
  }
  return String(Number(v.toPrecision(6)));
}

/** An operand: negative values in parentheses, so "a − b" never prints as "a − −b". */
const op = (s: string) => (s.startsWith('-') ? `(${s})` : s);

/**
 * Decimals that resolve a difference `d` to `sig` significant digits (for operands printed with
 * the same decimals, so the printed difference is the difference of the printed operands).
 */
export function decimalsFor(d: number, sig = 4, max = 10): number {
  if (!(Math.abs(d) > 0) || !Number.isFinite(d)) return 4;
  return Math.max(0, Math.min(max, sig - 1 - Math.floor(Math.log10(Math.abs(d)))));
}

const tv = (x: readonly number[] | null | undefined, d = 4) =>
  x ? `(${x.map((v) => tn(v, d)).join(',\\ ')})` : '\\text{—}';

/** A vector with per-coordinate fixed decimals. */
const tvf = (x: readonly number[], dec: readonly number[]) =>
  `(${x.map((v, j) => v.toFixed(dec[j])).join(',\\ ')})`;

/**
 * A computed number as UI text, with the same digits as `tn` in the formula above it (so a value
 * reads the same in the step's formula and in its quantity grid): `digits` significant digits,
 * ×10ⁿ below 10⁻³ and at or above 10⁵, trailing zeros kept.
 */
export function tnText(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(digits - 1).split('e');
    return `${m.replace('-', '−')}×10${superscript(Number(e))}`;
  }
  const r = v.toPrecision(digits);
  return r.includes('e') ? tnText(Number(r), digits) : r.replace('-', '−');
}

/** Significant digits of the temperature, in the formula and in the grid. */
export const T_DIGITS = 4;
/** Significant digits of the acceptance probability pₖ, in the formula and in the grid. */
export const P_DIGITS = 3;

/**
 * The Metropolis test of annealing and basin hopping, one short relation per row so it fits the
 * Details column at full size: Δf with operands that reproduce it digit for digit, the exponent
 * Δf/T computed on its own row, then pₖ = e^{−Δf/T} and its value. Downhill: Δf ≤ 0 and pₖ = 1.
 */
export function metropolisTex(
  fNew: number,
  fOld: number,
  T: number,
  p: number,
  newTex: string,
  tTex: string,
): string {
  const d = fNew - fOld;
  const big = Math.max(Math.abs(fNew), Math.abs(fOld));
  // Both operands with the decimals that resolve Δf to 4 digits, when that stays readable;
  // the printed Δf is then the difference of the printed operands. Else Δf alone.
  const dec = decimalsFor(d, 4);
  const fixed = big < 1e5 && big.toFixed(dec).length <= 14;
  const a = fNew.toFixed(dec),
    b = fOld.toFixed(dec);
  const dShown = fixed ? (Number(a) - Number(b)).toFixed(dec) : tn(d);
  const diff = fixed ? `${op(a)} - ${op(b)} = ${dShown}` : dShown;
  const head = `\\begin{aligned}\\Delta f &= ${newTex} - f(\\mathbf{x}_{k-1})\\\\ `;
  if (d <= 0) return `${head}&= ${diff} \\le 0\\\\ p_k &= 1\\end{aligned}`;
  const q = d / T;
  // A quotient's operands in ×10ⁿ form are parenthesized: a/(9.979×10⁻⁴), never a/9.979×10⁻⁴.
  const paren = (t: string) => (t.includes('\\times') ? `(${t})` : t);
  return (
    `${head}&= ${diff}\\\\ ` +
    `\\Delta f/${tTex} &= ${paren(dShown)}/${paren(tn(T, T_DIGITS))} = ${tn(q)}\\\\ ` +
    `p_k &= e^{-\\Delta f/${tTex}} = e^{-${tn(q)}}\\\\ ` +
    `&\\approx ${tn(p, P_DIGITS)}\\end{aligned}`
  );
}

export interface StepTexInput {
  method: string;
  trace: readonly Step[];
  k: number;
  params: Record<string, unknown>;
  /** Problem dimension n (for the CMA-ES constants). */
  n: number;
}

/**
 * A grid number: 5 significant digits, ×10ⁿ outside [10⁻³, 10⁵). A value that the digits
 * represent exactly prints short (3.3); a rounded one keeps its trailing zeros (1.0000), so
 * it never looks exact.
 */
function num(v: number): string {
  if (!Number.isFinite(v) || v === 0) return sig(v, 5);
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    if (Number(v.toExponential(2)) === v) return sci(v, 3);
    // sci() trims zeros; keep them: 2.90×10⁻⁷ for a rounded 2.9003×10⁻⁷.
    const [m] = v.toExponential(2).split('e');
    return sci(v, 3).replace(/^[^×]*/, m.replace('-', '−'));
  }
  return Number(v.toPrecision(5)) === v ? sig(v, 5) : v.toPrecision(5).replace('-', '−');
}

/** A value of the quantity grid (numbers and vectors as `num`). */
export function show(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number') return num(v);
  if (Array.isArray(v)) return `(${(v as number[]).map(num).join(', ')})`;
  return String(v);
}

export interface Quantity {
  tex: string;
  value: string;
  /** Spans two cells (vectors). */
  wide?: boolean;
}

/**
 * The live quantities of a step with labels that say exactly what they are (𝐱ₖ is the chain
 * state of annealing and basin hopping, 𝐠 the swarm's best, 𝐦ₖ the CMA-ES mean; f_best is the
 * best value found so far, which Step.fun records).
 */
export function stepQuantities(method: string, s: Step | undefined, popSize?: number): Quantity[] {
  if (!s) return [];
  const i = s.info;
  const k: Quantity = { tex: 'k', value: int(s.k) };
  const fBest: Quantity = { tex: 'f_{\\mathrm{best}}', value: show(s.fun) };
  switch (method) {
    case 'simulated_annealing':
      return [
        k,
        { tex: 'T_k', value: tnText(i.temperature as number | null, T_DIGITS) },
        fBest,
        { tex: '\\mathbf{x}_k', value: show(i.current), wide: true },
        { tex: 'f(\\mathbf{x}_k)', value: show(i.current_f) },
        { tex: 'f(\\mathbf{y})', value: show(i.candidate_f) },
        { tex: 'p_k', value: tnText(i.accept_prob as number | null, P_DIGITS) },
        { tex: '\\|\\mathbf{y} - \\mathbf{x}_{k-1}\\|', value: show(s.stepSize) },
      ];
    case 'particle_swarm':
      return [
        k,
        { tex: 'f(\\mathbf{g})', value: show(s.fun) },
        { tex: '\\max_i \\|\\mathbf{v}_i\\|', value: show(s.stepSize) },
        { tex: '\\mathbf{g}', value: show(s.x), wide: true },
        {
          tex: '\\max_i \\|\\mathbf{x}_i - \\mathbf{g}\\|_\\infty',
          value: show(searchScale(method, s)),
          wide: true,
        },
      ];
    case 'differential_evolution':
      return [
        k,
        fBest,
        {
          tex: '\\text{won}',
          value:
            s.k === 0
              ? '—'
              : `${acceptedCount(s)} / ${popSize ?? (i.population as unknown[]).length}`,
        },
        { tex: '\\mathbf{x}_{\\mathrm{best}}', value: show(s.x), wide: true },
        { tex: '\\text{spread}', value: show(searchScale(method, s)) },
      ];
    case 'cma_es': {
      const ps = i.p_sigma as Vector;
      return [
        k,
        { tex: '\\sigma_{k+1}', value: show(i.sigma) },
        fBest,
        { tex: '\\mathbf{m}_{k+1}', value: show(i.mean), wide: true },
        { tex: '\\|\\mathbf{p}_\\sigma\\|', value: show(Math.hypot(...ps)) },
        { tex: 'h_\\sigma', value: show(i.h_sigma) },
        { tex: '\\sigma\\sqrt{\\max_i C_{ii}}', value: show(searchScale(method, s)) },
      ];
    }
    case 'basin_hopping':
      return [
        k,
        fBest,
        { tex: '\\text{stall}', value: show(i.stall) },
        { tex: '\\mathbf{x}_k', value: show(i.current), wide: true },
        { tex: 'f(\\mathbf{x}_k)', value: show(i.current_f) },
        { tex: 'f(\\mathbf{z})', value: show(i.local_f) },
        { tex: 'p_k', value: tnText(i.accept_prob as number | null, P_DIGITS) },
        { tex: '\\#\\,\\text{minima}', value: show((i.minima as unknown[]).length) },
      ];
    default:
      return [k, fBest];
  }
}

/**
 * At the last CMA-ES step: when the mean settled far from the best sample (another basin), say
 * so — Result.x is the best sample ever evaluated, not the final mean.
 */
function splitNote(s: Step, mean: Vector, last: boolean): string {
  if (!last) return '';
  const best = s.x as Vector;
  const gap = Math.max(...mean.map((v, j) => Math.abs(v - best[j])));
  const scale = searchScale('cma_es', s) ?? 0;
  if (!(gap > 1e-6 + 100 * scale)) return '';
  return ` The mean settled at ${vec(mean, 3)}, but the best sample ever drawn is ${vec(best, 3)}: the result is the best sample (Hansen 2016), and the mean shows where the distribution went.`;
}

/** A tolerance in TeX for a note, e.g. $10^{-8}$ or $2.5\times 10^{-6}$. */
function texSci(v: number): string {
  return `$${tp(v)}$`;
}

/** Same rule as basin hopping's minima catalog: ‖z − m‖∞ ≤ 10⁻⁴(1 + ‖m‖∞). */
export function sameMinimum(z: Vector, m: Vector): boolean {
  let d = 0,
    a = 0;
  for (let j = 0; j < m.length; j++) {
    d = Math.max(d, Math.abs(z[j] - m[j]));
    a = Math.max(a, Math.abs(m[j]));
  }
  return d <= 1e-4 * (1 + a);
}

/**
 * The CMA-ES mean update with numbers that add up as printed: per coordinate, the decimals that
 * resolve the shift σₖ⟨𝐲⟩_w to 3 digits (and the mean to 4), and the printed shift is the
 * difference of the printed means. Beyond 5 decimals: the shift alone and the rounded mean.
 */
export function meanSum(m0: Vector, m1: Vector): string {
  const dec = m1.map((v, j) => {
    const shift = v - m0[j];
    const forMean = Math.max(0, 3 - Math.floor(Math.log10(Math.max(Math.abs(v), 1e-300))));
    return Math.max(decimalsFor(shift, 3), Math.min(forMean, 10));
  });
  // Beyond 5 decimals the sum no longer fits a line: give the shift itself (3 digits) and the
  // rounded new mean, marked ≈.
  const readable = dec.every((d) => d <= 5) && m0.concat(m1).every((v) => Math.abs(v) < 1e5);
  if (!readable) {
    const shift = m1.map((v, j) => v - m0[j]);
    return `&= \\mathbf{m}_k + ${tv(shift, 3)}\\\\ &\\approx ${tv(m1)}`;
  }
  const a = m0.map((v, j) => Number(v.toFixed(dec[j])));
  const b = m1.map((v, j) => Number(v.toFixed(dec[j])));
  const shift = b.map((v, j) => v - a[j]);
  // A column addition: 𝐦ₖ, then + the shift, then 𝐦ₖ₊₁ (each line fits a phone).
  return `&= ${tvf(m0, dec)}\\\\ &\\phantom{=}\\, + ${tvf(shift, dec)}\\\\ &= ${tvf(m1, dec)}`;
}

/** The instantiated rule, or null when the step has no update (k = 0). */
export function stepTex({
  method,
  trace,
  k,
  params,
  n,
}: StepTexInput): { tex: string; note: string } | null {
  const s = trace[k];
  if (!s) return null;
  const i = s.info;
  if (k === 0) return null;
  const prev = trace[k - 1].info;
  switch (method) {
    case 'simulated_annealing': {
      const T = i.temperature as number;
      const fx = prev.current_f as number;
      if (i.inside === false)
        return {
          tex: `\\mathbf{y} = ${tv(i.candidate as Vector)} \\notin \\Omega`,
          note: `The proposal left the box: it is rejected without an evaluation of $f$ ($T_k = ${tn(T, T_DIGITS)}$).`,
        };
      const fy = i.candidate_f as number;
      return {
        tex: metropolisTex(fy, fx, T, i.accept_prob as number, 'f(\\mathbf{y})', 'T_k'),
        note: i.accepted
          ? 'The Metropolis test accepted the proposal: the walker moved to $\\mathbf{y}$.'
          : 'The Metropolis test rejected the proposal ($u \\ge p_k$): the walker stayed.',
      };
    }
    case 'particle_swarm': {
      const X = i.particles as Vector[];
      const g = i.global_best as Vector;
      let spread = 0;
      for (const x of X) spread = Math.max(spread, Math.abs(x[0] - g[0]), Math.abs(x[1] - g[1]));
      const w = Number(params.w),
        c1 = Number(params.c1),
        c2 = Number(params.c2);
      return {
        tex:
          `\\begin{gathered}w = ${tp(w)},\\quad ${c1 === c2 ? `c_1 = c_2 = ${tp(c1)}` : `c_1 = ${tp(c1)},\\ c_2 = ${tp(c2)}`}\\\\ ` +
          `\\max\\nolimits_i \\|\\mathbf{x}_i - \\mathbf{g}\\|_\\infty = ${tn(spread)}\\end{gathered}`,
        note: `The swarm has collapsed when this spread is at most ${texSci(Number(params.xtol))} and every $f(\\mathbf{x}_i)$ is within ${texSci(Number(params.ftol))} of $f(\\mathbf{g})$.`,
      };
    }
    case 'differential_evolution': {
      const old = prev.population as Vector[];
      const oldF = prev.population_f as number[];
      const trials = i.trials as Vector[];
      const j = deHighlight(old, i.mutants as Vector[], trials, k % trials.length);
      const fu = (i.trials_f as number[])[j];
      const acc = (i.accepted as boolean[])[j];
      const won = acceptedCount(s);
      return {
        tex:
          `\\begin{aligned}\\mathbf{u}_{${j}} &= ${tv(trials[j])}\\\\ ` +
          `f(\\mathbf{u}_{${j}}) &= ${tn(fu)} ${acc ? '\\le' : '>'} f(\\mathbf{x}_{${j}}) = ${tn(oldF[j])}\\end{aligned}`,
        note: `Target ${j} is drawn in full: ${acc ? 'its trial replaced it' : 'it kept its place'}. In this generation, ${won} of ${trials.length} ${trials.length === 1 ? 'trial' : 'trials'} won.`,
      };
    }
    case 'cma_es': {
      const m0 = i.sample_mean as Vector;
      const m1 = i.mean as Vector;
      const s0 = i.sample_sigma as number;
      const s1 = i.sigma as number;
      const c = cmaConstants(n, Number(params.pop_size ?? 0));
      const ps = i.p_sigma as Vector;
      const norm = Math.hypot(...ps);
      return {
        tex:
          `\\begin{aligned}\\mathbf{m}_{k+1} &= \\mathbf{m}_k + \\sigma_k \\langle \\mathbf{y} \\rangle_w\\\\ ` +
          `${meanSum(m0, m1)}\\\\ ` +
          `\\sigma_{k+1} &= \\sigma_k\\, e^{(c_\\sigma/d_\\sigma)(\\|\\mathbf{p}_\\sigma\\|/\\mathbb{E}\\|\\mathcal{N}(0,I)\\| - 1)}\\\\ ` +
          `&= ${tn(s0)}\\, e^{${tn(c.cSigma / c.dSigma)}\\,(${tn(norm)}/${tn(c.chiN)} - 1)} = ${tn(s1)}\\end{aligned}`,
        note:
          (norm > c.chiN
            ? '$\\|\\mathbf{p}_\\sigma\\|$ is longer than its expected length under random selection: consecutive steps agree, so $\\sigma$ grows.'
            : '$\\|\\mathbf{p}_\\sigma\\|$ is shorter than its expected length under random selection: steps cancel, so $\\sigma$ shrinks.') +
          splitNote(s, m1, trace.length - 1 === k),
      };
    }
    case 'basin_hopping': {
      const fz = i.local_f as number;
      const fx = prev.current_f as number;
      const T = Number(params.T ?? 1);
      const nMin = (i.minima as unknown[]).length;
      const nPrev = (prev.minima as unknown[]).length;
      const stall = Number(i.stall);
      const same = sameMinimum(i.local_min as Vector, prev.current as Vector);
      const found = nMin > nPrev ? ' It is a new minimum.' : '';
      const what = i.accepted
        ? same
          ? 'Accepted, but the descent returned to the current minimum: the walk stays in place.'
          : `Accepted: the walk moved to ${nMin > nPrev ? 'the new' : 'a known'} minimum $\\mathbf{z}$.`
        : `Rejected: the walk stays at its minimum.${found}`;
      return {
        tex: metropolisTex(fz, fx, T, i.accept_prob as number, 'f(\\mathbf{z})', 'T'),
        note: `${what} So far, the walk found ${nMin} distinct ${nMin === 1 ? 'minimum' : 'minima'}; the best value has not improved for ${stall} ${stall === 1 ? 'hop' : 'hops'}.`,
      };
    }
    default:
      return null;
  }
}

/** Columns of the quantity grid (GlobalLab.module.css `.live`). */
const COLS = 3;

/**
 * Grid spans with no empty cell: a wide cell takes 2 columns; a cell before a wide cell that
 * does not fit its row, and the last cell, stretch to the end of their row.
 */
export function cellSpans(wide: readonly boolean[], cols = COLS): number[] {
  const spans = wide.map((w) => (w ? Math.min(2, cols) : 1));
  let col = 0;
  spans.forEach((s, i) => {
    if (col + s > cols) {
      spans[i - 1] += cols - col;
      col = 0;
    }
    col = (col + s) % cols;
  });
  if (col !== 0 && spans.length) spans[spans.length - 1] += cols - col;
  return spans;
}

/**
 * Steps whose "This step" text is the tallest of its kind in the trace: for each number of
 * formula rows, the step with the longest note (and k = 0, which has no formula). Rendered
 * hidden in the same grid cell as the current step, they hold the block at the height of the
 * tallest step of the run, so the panel never jumps while the trace plays.
 */
export function layoutSamples(input: Omit<StepTexInput, 'k'>): number[] {
  const best = new Map<number, { k: number; score: number }>();
  for (let k = 1; k < input.trace.length; k++) {
    const r = stepTex({ ...input, k });
    if (!r) continue;
    const rows = (r.tex.match(/\\\\/g) ?? []).length;
    const score = r.note.length * 4 + r.tex.length;
    const seen = best.get(rows);
    if (!seen || score > seen.score) best.set(rows, { k, score });
  }
  return [0, ...[...best.values()].map((b) => b.k)];
}
