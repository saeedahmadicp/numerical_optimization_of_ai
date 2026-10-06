/**
 * "This step": the update of step g (𝐱_{g−1} → 𝐱_g) with its numbers filled in, as KaTeX, one
 * plain sentence on what the method decided, and the step's quantities. Everything is read from
 * Step.info (keys of src/numopt/unconstrained/*.py) or recomputed from the problem's exact f, ∇f.
 */
import type { Problem2D, Step } from '../../core/types';
import { int, sci, sig, vec } from '../../core/format';
import type { Kind } from './catalog';
import { adaptiveMultiplicand, isVec, len, nearestMinimizer, rayOf } from './geometry';

/** A number in TeX: 4 significant digits, ×10ⁿ below 10⁻³ and from 10⁵ on. */
export function tn(x: unknown, digits = 4): string {
  if (typeof x !== 'number' || Number.isNaN(x)) return '\\text{—}';
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : '-\\infty';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a < 1e-3 || a >= 1e5) {
    const [mant, e] = x.toExponential(digits - 1).split('e');
    const mm = mant.replace(/\.?0+$/, '');
    return `${mm === '1' ? '' : mm === '-1' ? '-' : `${mm}\\times `}10^{${Number(e)}}`;
  }
  return String(Number(x.toPrecision(digits)));
}

/** A 2-vector in TeX: (a,\ b). */
export const tv = (x: unknown, d = 4): string =>
  Array.isArray(x) ? `(${(x as unknown[]).map((c) => tn(c, d)).join(',\\ ')})` : '\\text{—}';

/** A 2-vector as a column (beside a matrix, as in a matrix–vector product). */
export const tc = (x: unknown, d = 3): string =>
  Array.isArray(x)
    ? `\\begin{pmatrix} ${(x as unknown[]).map((c) => tn(c, d)).join(' \\\\ ')} \\end{pmatrix}`
    : '\\text{—}';

/** A 2×2 matrix in TeX. */
export const tm = (M: unknown, d = 4): string =>
  Array.isArray(M) && M.length === 2
    ? `\\begin{pmatrix} ${(M as unknown[][]).map((r) => r.map((c) => tn(c, d)).join(' & ')).join(' \\\\ ')} \\end{pmatrix}`
    : '\\text{—}';

/** A value as the quantity grid shows it. */
export function show(x: unknown): string {
  if (x === null || x === undefined) return '—';
  if (typeof x === 'boolean') return x ? 'yes' : 'no';
  if (typeof x === 'number')
    return x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5) ? sci(x, 3) : sig(x, 5);
  if (Array.isArray(x) && x.every((c) => typeof c === 'number')) return vec(x as number[], 4);
  return String(x);
}

export interface Quantity {
  tex: string;
  value: string;
  wide?: boolean;
}

/**
 * A strip of one number per step (the schedule hₖ, FISTA's βₖ, ARC's σₖ, …) drawn under the
 * step's heading, with the current step highlighted: the history and the plan of the run at a
 * glance. Values are indexed by k (index 0 is the start and is not drawn).
 */
export interface BarStrip {
  kind: 'bars';
  /** TeX of the quantity, e.g. `h_{k-1}`. */
  label: string;
  values: (number | null)[];
  log: boolean;
  /** A dashed reference level (e.g. h = 2) and its TeX. */
  ref?: { value: number; tex: string };
  /** Steps marked under their bar (certified checkpoints, restarts, rejected steps). */
  marks: number[];
  /** What a mark means, in words. */
  markLabel?: string;
}

/** AA's weights c* on the history window x_{k−1−m}, …, x_{k−1}, as signed bars. */
export interface WeightStrip {
  kind: 'weights';
  weights: number[];
  /** TeX names of the iterates the weights multiply. */
  names: string[];
}

export type StripSpec = BarStrip | WeightStrip;

export interface StepView {
  /** Display TeX (null at k = 0). */
  tex: string | null;
  note: string;
  quantities: Quantity[];
  /** A strip over the run (or the step's weights), drawn at a fixed height. */
  strip?: StripSpec;
}

export interface StepInput {
  kind: Kind;
  method: string;
  trace: readonly Step[];
  g: number;
  params: Readonly<Record<string, unknown>>;
  problem: Problem2D;
}

const X = (i: number | string) => `\\mathbf{x}_{${i}}`;
const GF = (i: number | string) => `\\nabla f(${X(i)})`;
const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : null);
const al = (lines: string[]) => `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`;

const SEARCH_NAME: Record<string, string> = {
  backtracking: 'Armijo backtracking',
  strong_wolfe: 'The strong Wolfe search',
  weak_wolfe: 'The weak Wolfe search',
  goldstein: 'The Goldstein search',
  exact_quadratic: 'The exact quadratic step',
};

/** "1 Cholesky attempt", "3 Cholesky attempts". */
export function plural(n: number, noun: string): string {
  return `${int(n)} ${noun}${n === 1 ? '' : 's'}`;
}

/**
 * Significant digits that show a shift τ = −a + β to the resolution of β (so that the printed
 * ∇²f + τI is visibly nonsingular): 4 at least, 10 at most.
 */
export function tauDigits(tau: number, beta: number): number {
  if (!(tau > 0) || !(beta > 0)) return 4;
  return Math.min(10, Math.max(4, Math.ceil(Math.log10(tau / beta)) + 1));
}

function searchNote(name: string, trials: number, alpha: number | null): string {
  if (alpha === null) return `${name} found no acceptable step.`;
  const n = trials === 1 ? 'one trial' : `${int(trials)} trials`;
  return `${name} accepted $\\alpha = ${tn(alpha)}$ after ${n}.`;
}

function base(s: Step, withGrad = true): Quantity[] {
  const q: Quantity[] = [
    { tex: 'k', value: int(s.k) },
    { tex: `f(${X('k')})`, value: show(s.fun) },
  ];
  if (withGrad) q.push({ tex: `\\|\\nabla f(${X('k')})\\|_2`, value: show(s.gradNorm) });
  q.push({ tex: X('k'), value: show(s.x), wide: true });
  return q;
}

const START_NOTE: Partial<Record<Kind, string>> = {
  tr: 'Step 0 is the start: $\\mathbf{x}_0$ and the first trust region, of radius $\\Delta_0$. Step forward to see the first model step.',
  nm: 'Step 0 is the start: the initial simplex at $\\mathbf{x}_0$. Step forward to see the first operation.',
  compass: 'Step 0 is the start: $\\mathbf{x}_0$ and the first poll step $\\Delta_0$.',
};

const cells = (q: readonly Quantity[]) => q.reduce((n, c) => n + (c.wide ? 2 : 1), 0);

/**
 * Complete the last row of the 3-column grid with the step's decrease f(𝐱ₖ₋₁) − f(𝐱ₖ) and its
 * length ‖𝐱ₖ − 𝐱ₖ₋₁‖, when there is room (they are meaningful for every method).
 */
function fill(v: StepView, i: StepInput): StepView {
  const s = i.trace[i.g];
  const prev = i.trace[i.g - 1];
  if (!s || !prev) return v;
  const q = [...v.quantities];
  const extra: Quantity[] = [];
  if (typeof s.fun === 'number' && typeof prev.fun === 'number')
    extra.push({ tex: `f_{${i.g - 1}} - f_{${i.g}}`, value: show(prev.fun - s.fun) });
  if (isVec(s.x) && isVec(prev.x))
    extra.push({
      tex: `\\|${X(i.g)} - ${X(i.g - 1)}\\|`,
      value: show(Math.hypot(s.x[0] - prev.x[0], s.x[1] - prev.x[1])),
    });
  for (const e of extra) if (cells(q) % 3 !== 0) q.push(e);
  return { ...v, quantities: q };
}

/** The rule of step g with its numbers, a note, the quantities and the strip of the run. */
export function stepView(i: StepInput): StepView {
  const v = fill(stepViewRaw(i), i);
  const strip = i.kind === 'anderson' ? weightsOf(i.trace, i.g) : stripOf(i);
  return strip ? { ...v, strip } : v;
}

/** AA's weights on the window of step g (empty for a plain gradient step and at k = 0). */
export function weightsOf(trace: readonly Step[], g: number): WeightStrip {
  const s = trace[g];
  const c = s && Array.isArray(s.info.coefficients) ? (s.info.coefficients as number[]) : [];
  const mk = s ? (num(s.info.memory) ?? 0) : 0;
  if (g < 1 || mk === 0) return { kind: 'weights', weights: [], names: [] };
  return {
    kind: 'weights',
    weights: c.slice(),
    names: c.map((_, j) => `\\mathbf{x}_{${g - 1 - mk + j}}`),
  };
}

const certifiedStep = (s: Step) =>
  typeof s.info.bound_f === 'number' || typeof s.info.bound_dist === 'number';

/** The strip of the run for the kinds that have one (it does not depend on g). */
export function stripOf(i: Pick<StepInput, 'kind' | 'trace'>): StripSpec | undefined {
  const t = i.trace;
  const at = (key: string) => t.map((s, k) => (k === 0 ? null : num(s.info[key])));
  const marks = (test: (s: Step) => boolean) => t.flatMap((s, k) => (k > 0 && test(s) ? [k] : []));
  switch (i.kind) {
    case 'schedule':
      return {
        kind: 'bars',
        label: 'h_{k-1}',
        values: at('h'),
        log: true,
        ref: { value: 2, tex: 'h = 2' },
        marks: marks(certifiedStep),
        markLabel: 'certified',
      };
    case 'ogm':
      return {
        kind: 'bars',
        label: '(\\theta_{k-1} - 1)/\\theta_k',
        values: t.map((s, k) => {
          const a = k > 0 ? num(t[k - 1].info.theta) : null;
          const b = num(s.info.theta);
          return a !== null && b !== null && b > 0 ? (a - 1) / b : null;
        }),
        log: false,
        marks: marks(certifiedStep),
        markLabel: 'certified',
      };
    case 'fista':
      return {
        kind: 'bars',
        label: '\\beta_k',
        values: at('beta'),
        log: false,
        marks: marks((s) => s.info.restarted === true),
        markLabel: 'restart',
      };
    case 'arc':
      return {
        kind: 'bars',
        label: '\\sigma_k',
        values: at('sigma'),
        log: true,
        marks: marks((s) => s.info.accepted === false),
        markLabel: 'rejected',
      };
    case 'regnewton':
      return {
        kind: 'bars',
        label: '\\lambda_k',
        values: at('lambda'),
        log: true,
        marks: marks((s) => s.info.pd === false),
        markLabel: '∇²f + λI indefinite',
      };
    default:
      return undefined;
  }
}

/** f(𝐱ₖ) − f⋆ and ‖𝐱₀ − 𝐱⋆‖² against the problem's nearest known minimizer. */
function certificateNumbers(i: StepInput): { gap: number; dist0: number; distK: number } | null {
  const last = i.trace[i.trace.length - 1];
  const star = nearestMinimizer(i.problem, last.x);
  const x0 = i.trace[0].x;
  const xk = i.trace[i.g].x;
  const fk = i.trace[i.g].fun;
  if (!star || !isVec(x0) || !isVec(xk) || typeof fk !== 'number') return null;
  const fStar = i.problem.f([star[0], star[1]]);
  return {
    gap: fk - fStar,
    dist0: (x0[0] - star[0]) ** 2 + (x0[1] - star[1]) ** 2,
    distK: (xk[0] - star[0]) ** 2 + (xk[1] - star[1]) ** 2,
  };
}

const RESTART_TEST: Record<string, string> = {
  gradient:
    'the gradient test fired: $\\nabla f(\\mathbf{y}_k)^{\\top}(\\mathbf{x}_k - \\mathbf{x}_{k-1}) > 0$, so the momentum pointed uphill',
  function: 'the function test fired: $f(\\mathbf{x}_k) > f(\\mathbf{x}_{k-1})$',
};

const ITERATION_CLASS: Record<string, string> = {
  very_successful:
    'very successful ($\\rho > \\eta_2$): σ decreases to $\\max(\\min(\\sigma, \\|\\nabla f\\|), \\varepsilon)$',
  successful: 'successful ($\\eta_1 \\le \\rho \\le \\eta_2$): σ stays',
  unsuccessful: 'unsuccessful ($\\rho < \\eta_1$): the step is rejected and σ grows by γ',
};

function stepViewRaw(i: StepInput): StepView {
  const s = i.trace[i.g];
  if (!s) return { tex: null, note: '', quantities: [] };
  const df = i.kind === 'nm' || i.kind === 'powell' || i.kind === 'hj' || i.kind === 'compass';
  const q = base(s, !df);
  if (i.g === 0)
    return {
      tex: null,
      note:
        START_NOTE[i.kind] ??
        'Step 0 is the start $\\mathbf{x}_0$ and its values. Step forward to see the first update.',
      quantities: q,
    };
  const k = i.g;
  const prev = i.trace[k - 1];
  const info = s.info;
  const P = i.params;
  switch (i.kind) {
    case 'line': {
      const ray = rayOf('line', i.trace, k);
      const a = num(info.alpha);
      const tex = al([
        `${X(k)} &= ${X(k - 1)} - \\alpha_{${k}}\\,${GF(k - 1)}`,
        `&= ${tv(prev.x)} + ${tn(a)}\\,${tv(info.direction)}`,
        `&= ${tv(s.x)}`,
      ]);
      let note: string;
      if (i.method === 'barzilai_borwein') {
        note =
          info.reset === true
            ? `The BB step was reset to $1/\\|${GF(k - 1)}\\| = ${tn(info.alpha_bb)}$.`
            : `$\\mathbf{s}^{\\top}\\mathbf{s}/\\mathbf{s}^{\\top}\\mathbf{y} = ${tn(info.bb1)}$ (BB1) and $\\mathbf{s}^{\\top}\\mathbf{y}/\\mathbf{y}^{\\top}\\mathbf{y} = ${tn(info.bb2)}$ (BB2); the ${P.variant === 'bb2' ? 'BB2' : 'BB1'} step was tried first.`;
        if (num(info.f_ref) !== null && ray)
          note += ` The nonmonotone test compares with the worst recent value $f_{\\mathrm{ref}} = ${tn(info.f_ref)}$, not $f(${X(k - 1)})$: ${ray.trials.length === 1 ? 'the first trial passed.' : `${ray.trials.length} trials.`}`;
      } else {
        const rule = String(P.step_rule ?? 'backtracking');
        note =
          rule === 'fixed'
            ? `A fixed step $\\alpha = ${tn(a)}$: no line search.`
            : rule === 'exact_quadratic'
              ? `The exact minimizer of $f$ along $-\\nabla f$: $\\alpha = -\\nabla f^{\\top}\\mathbf{p} / \\mathbf{p}^{\\top}\\nabla^2 f\\,\\mathbf{p} = ${tn(a)}$. The step ends where the ray touches a level set of $f$.`
              : searchNote(SEARCH_NAME[rule] ?? rule, ray?.trials.length ?? 0, a);
      }
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: `\\alpha_{${k}}`, value: show(a) },
          { tex: '\\#\\,\\text{trials}', value: int(ray?.trials.length ?? 0) },
        ],
      };
    }
    case 'coord': {
      const c = num(info.coordinate);
      const name = c === 0 ? 'x' : 'y';
      const tex = al([
        `${X(k)} &= ${X(k - 1)} + \\alpha_{${k}}\\,\\mathbf{p}_{${k}},\\quad \\mathbf{p}_{${k}} = ${tv(info.direction)}`,
        `&= ${tv(s.x)}`,
      ]);
      const note =
        info.newton === true
          ? `A 1-D Newton step on the coordinate $${name}$: $\\partial f/\\partial ${name}$ divided by $\\partial^2 f/\\partial ${name}^2 = ${tn(info.curvature)}$.`
          : info.newton === false && num(info.alpha) === 0
            ? `The coordinate $${name}$ was skipped: its step would not change $\\mathbf{x}$.`
            : `$\\partial^2 f/\\partial ${name}^2 = ${tn(info.curvature)} \\le 0$, so a gradient step on $${name}$ with backtracking replaced the Newton step.`;
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: 'i', value: c === null ? '—' : `${c + 1} (${name})` },
          { tex: `\\partial^2_{ii} f`, value: show(info.curvature) },
          { tex: `\\alpha_{${k}}`, value: show(info.alpha) },
        ],
      };
    }
    case 'heavy': {
      const vPrev = prev.info.velocity;
      const tex = al([
        `\\mathbf{v}_{${k}} &= \\beta\\,\\mathbf{v}_{${k - 1}} - \\alpha\\,${GF(k - 1)}`,
        `&= ${tn(P.beta)}\\,${tv(vPrev)} - ${tn(P.lr)}\\,${tv(prev.info.grad)}`,
        `&= ${tv(info.velocity)}, \\qquad ${X(k)} = ${X(k - 1)} + \\mathbf{v}_{${k}}`,
      ]);
      const vp = isVec(vPrev) ? len(vPrev) : 0;
      const vk = isVec(info.velocity) ? len(info.velocity) : 0;
      return {
        tex,
        note: `The ball keeps the fraction $\\beta = ${tn(P.beta)}$ of its last velocity and adds the gradient step: $\\|\\mathbf{v}\\|$ went from $${tn(vp)}$ to $${tn(vk)}$.`,
        quantities: [...q, { tex: `\\|\\mathbf{v}_{${k}}\\|`, value: show(vk) }],
      };
    }
    case 'nesterov': {
      const tex = al([
        `\\mathbf{y} &= ${X(k - 1)} + \\mu\\,\\mathbf{v}_{${k - 1}} = ${tv(info.lookahead)}`,
        `\\mathbf{v}_{${k}} &= \\mu\\,\\mathbf{v}_{${k - 1}} - \\alpha\\,\\nabla f(\\mathbf{y}) = ${tv(info.velocity)}`,
        `${X(k)} &= ${X(k - 1)} + \\mathbf{v}_{${k}} = ${tv(s.x)}`,
      ]);
      return {
        tex,
        note: `The gradient is taken at the look-ahead point $\\mathbf{y}$, where momentum is about to carry the iterate: $\\nabla f(\\mathbf{y}) = ${tv(info.grad_lookahead)}$.`,
        quantities: [...q, { tex: '\\mathbf{y}', value: show(info.lookahead), wide: true }],
      };
    }
    case 'adaptive': {
      const u = adaptiveMultiplicand(i.method, i.trace, k);
      const uName =
        i.method === 'adam' || i.method === 'adamw'
          ? '\\hat{\\mathbf{m}}_{' + k + '}'
          : i.method === 'nadam'
            ? '\\bar{\\mathbf{m}}_{' + k + '}'
            : i.method === 'adamax' || i.method === 'amsgrad'
              ? '\\mathbf{m}_{' + k + '}'
              : GF(k - 1);
      const decay = i.method === 'adamw' ? ` + ${tv(info.decay)}` : '';
      const tex = al([
        `${X(k)} &= ${X(k - 1)} - \\mathbf{D}_{${k}} \\odot ${uName}${i.method === 'adamw' ? ` - \\lambda\\,${X(k - 1)}` : ''}`,
        `&= ${tv(prev.x)} - ${tv(info.lr_eff, 3)} \\odot ${tv(u, 3)}${decay}`,
        `&= ${tv(s.x)}`,
      ]);
      const lr = info.lr_eff;
      const ratio =
        isVec(lr) && lr[0] > 0 && lr[1] > 0
          ? Math.max(lr[0], lr[1]) / Math.min(lr[0], lr[1])
          : null;
      return {
        tex,
        note:
          `$\\mathbf{D}_{${k}} = ${tv(lr, 3)}$ is the per-coordinate step` +
          (ratio !== null
            ? `: one coordinate moves $${tn(ratio, 3)}$ times as far per unit of ${uName.startsWith('\\nabla') ? 'gradient' : 'moment'} as the other, so the step turns away from $-${uName}$.`
            : '.'),
        quantities: [
          ...q,
          { tex: '\\alpha', value: show(P.lr) },
          { tex: `\\mathbf{D}_{${k}}`, value: show(lr), wide: true },
        ],
      };
    }
    case 'newton': {
      const tau = num(info.tau);
      const H = prev.info.hess;
      const steep = info.direction_type === 'steepest';
      // τ = −min aᵢᵢ + β (doubled until Cholesky succeeds) exceeds −λ_min by about β, so τ and
      // ∇²f are printed with enough digits to show it: at 4 digits ∇²f + τI would look singular.
      const shiftDigits = tau !== null && tau > 0 ? tauDigits(tau, num(P.beta) ?? 1e-3) : 3;
      const lhs =
        tau !== null && tau > 0
          ? `\\left(${tm(H, shiftDigits)} + ${tn(tau, shiftDigits)}\\,I\\right)`
          : tm(H, 3);
      const tex = steep
        ? al([
            `\\mathbf{p}_{${k}} &= -${GF(k - 1)} = ${tv(info.direction)}`,
            `${X(k)} &= ${X(k - 1)} + ${tn(info.alpha)}\\,\\mathbf{p}_{${k}} = ${tv(s.x)}`,
          ])
        : // Aligned at the left edge, not at "=": the wide left side of the system and the wide
          // right side of the update would otherwise add up to one very wide formula.
          al([
            `&${lhs}\\,\\mathbf{p}_{${k}} = -${tc(prev.info.grad)}`,
            `&\\mathbf{p}_{${k}} = ${tv(info.direction)}`,
            `&${X(k)} = ${X(k - 1)} + ${tn(info.alpha)}\\,\\mathbf{p}_{${k}} = ${tv(s.x)}`,
          ]);
      const eig = prev.info.hess_eigs;
      const def =
        Array.isArray(eig) && typeof eig[0] === 'number'
          ? eig[0] > 0
            ? 'positive definite'
            : (eig[eig.length - 1] as number) < 0
              ? 'negative definite'
              : 'indefinite'
          : null;
      let note = def
        ? `$\\nabla^2 f(${X(k - 1)})$ has the eigenvalues $${tv(eig, 3)}$: ${def}. `
        : '';
      if (i.method === 'pure_newton')
        note +=
          info.descent === false
            ? 'The Newton step goes uphill ($\\nabla f^{\\top}\\mathbf{p} > 0$): pure Newton takes it anyway.'
            : 'Pure Newton jumps to the stationary point of the quadratic model ($\\alpha = 1$).';
      else if (steep)
        note += 'The Newton direction failed the descent test, so $-\\nabla f$ was used.';
      else if (tau !== null && tau > 0)
        note += `The shift $\\tau = ${tn(tau, shiftDigits)}$ made $\\nabla^2 f + \\tau I$ positive definite (${plural(num(info.chol_attempts) ?? 0, 'Cholesky attempt')}). `;
      if (i.method !== 'pure_newton') {
        const ray = rayOf('newton', i.trace, k);
        const rule = String(P.line_search ?? 'backtracking');
        note +=
          ' ' +
          searchNote(
            `${SEARCH_NAME[rule] ?? 'The line search'} from $\\alpha = 1$`,
            ray?.trials.length ?? 0,
            num(info.alpha),
          );
      }
      return {
        tex,
        note: note.trim(),
        quantities: [
          ...q,
          { tex: `\\alpha_{${k}}`, value: show(info.alpha) },
          ...(tau !== null ? [{ tex: '\\tau', value: show(tau) }] : []),
          { tex: `\\lambda(\\nabla^2 f)`, value: show(prev.info.hess_eigs), wide: tau === null },
        ],
      };
    }
    case 'qn': {
      const H =
        info.reset === true
          ? [
              [1, 0],
              [0, 1],
            ]
          : prev.info.H;
      const tex = al([
        // The matrix–vector product has its own row: with 𝐩ₖ's definition on the same row it
        // is too wide for the Details column at full size.
        `\\mathbf{p}_{${k}} &= -H_{${k - 1}}\\,${GF(k - 1)}`,
        `&= -${tm(H, 3)}${tc(prev.info.grad)}`,
        `&= ${tv(info.direction, 3)}`,
        `${X(k)} &= ${X(k - 1)} + ${tn(info.alpha, 3)}\\,\\mathbf{p}_{${k}} = ${tv(s.x)}`,
        `\\mathbf{y}^{\\top}\\mathbf{s} &= ${tn(info.curvature, 3)}\\ \\Rightarrow\\ \\text{update ${info.update === 'skipped' ? 'skipped' : 'applied'}}`,
      ]);
      const ray = rayOf('qn', i.trace, k);
      let note = searchNote(
        SEARCH_NAME[String(P.line_search ?? 'strong_wolfe')] ?? 'The line search',
        ray?.trials.length ?? 0,
        num(info.alpha),
      );
      if (info.reset === true)
        note = '$H$ was reset to $I$: $-H\\nabla f$ failed the descent test. ' + note;
      if (num(info.gamma) !== null && i.method !== 'lbfgs')
        note += ` $H_0$ was first scaled by $\\gamma = \\mathbf{y}^{\\top}\\mathbf{s}/\\mathbf{y}^{\\top}\\mathbf{y} = ${tn(info.gamma)}$.`;
      note +=
        info.update === 'skipped'
          ? ' The pair $(\\mathbf{s}, \\mathbf{y})$ failed the curvature test, so $H$ is unchanged.'
          : ' The new $H_k$ satisfies the secant equation $H_k\\mathbf{y} = \\mathbf{s}$.';
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: `\\alpha_{${k}}`, value: show(info.alpha) },
          { tex: '\\mathbf{y}^{\\top}\\mathbf{s}', value: show(info.curvature) },
          ...(i.method === 'lbfgs' ? [{ tex: 'm_k', value: show(info.memory) }] : []),
        ],
      };
    }
    case 'cg': {
      const beta = prev.info.beta;
      const restart = prev.info.restart;
      // 𝐝ₖ₋₁ is the direction searched in step k; at k = 1 (and on a restart) it is −∇f.
      const dLine =
        k === 1 || !(num(beta) !== null && beta !== 0)
          ? `\\mathbf{d}_{${k - 1}} &= -${GF(k - 1)}${k === 1 ? '' : `\\quad (\\beta_{${k - 1}} = 0)`}`
          : `\\mathbf{d}_{${k - 1}} &= -${GF(k - 1)} + \\beta_{${k - 1}}\\,\\mathbf{d}_{${k - 2}},\\quad \\beta_{${k - 1}} = ${tn(beta)}`;
      const tex = al([
        dLine,
        `&= ${tv(prev.info.direction)}`,
        `${X(k)} &= ${X(k - 1)} + ${tn(info.alpha)}\\,\\mathbf{d}_{${k - 1}} = ${tv(s.x)}`,
      ]);
      const why: Record<string, string> = {
        initial: 'The first direction is $-\\nabla f$ ($\\beta = 0$).',
        periodic: 'A periodic restart after $n$ steps: $\\beta = 0$.',
        powell: `Powell's restart test fired: $|\\nabla f_k^{\\top}\\nabla f_{k-1}|/\\|\\nabla f_k\\|^2 = ${tn(prev.info.powell_ratio, 3)} \\ge 0.1$, so $\\beta = 0$.`,
        breakdown: '$\\beta$ was undefined, so the method restarted with $-\\nabla f$.',
        not_descent:
          'The CG direction was not downhill, so the method restarted with $-\\nabla f$.',
      };
      const ray = rayOf('cg', i.trace, k);
      const note =
        (typeof restart === 'string'
          ? (why[restart] ?? `Restart (${restart}).`)
          : `$\\beta = ${tn(beta)}$ bends $-\\nabla f$ toward the previous direction.`) +
        ' ' +
        searchNote(
          SEARCH_NAME[String(P.line_search ?? 'strong_wolfe')] ?? 'The line search',
          ray?.trials.length ?? 0,
          num(info.alpha),
        );
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: `\\beta_{${k - 1}}`, value: show(beta) },
          { tex: `\\alpha_{${k}}`, value: show(info.alpha) },
          { tex: '\\text{restart}', value: typeof restart === 'string' ? restart : '—' },
        ],
      };
    }
    case 'tr': {
      const accepted = info.accepted === true;
      const tex = al([
        `\\rho_{${k}} &= \\frac{f(${X(k - 1)}) - f(${X(k - 1)} + \\mathbf{p})}{m(\\mathbf{0}) - m(\\mathbf{p})} = \\frac{${tn(info.actual)}}{${tn(info.predicted)}} = ${tn(info.rho, 3)}`,
        `\\Delta &: ${tn(info.radius)} \\to ${tn(info.new_radius)},\\qquad ${X(k)} = ${accepted ? `${X(k - 1)} + \\mathbf{p}` : X(k - 1)}`,
      ]);
      const r = num(info.rho);
      const eta = num(P.eta) ?? 0.15;
      let note = accepted
        ? `$\\rho = ${tn(r, 3)} > \\eta = ${tn(eta)}$: the step is accepted`
        : `$\\rho = ${tn(r, 3)} \\le \\eta = ${tn(eta)}$: the model mispredicted this step, so it is rejected`;
      note +=
        num(info.new_radius) !== null && num(info.radius) !== null
          ? (info.new_radius as number) > (info.radius as number)
            ? ' and the region doubles.'
            : (info.new_radius as number) < (info.radius as number)
              ? ' and the region shrinks to $\\Delta/4$.'
              : ' and $\\Delta$ stays.'
          : '.';
      const extra: Quantity[] = [];
      if (i.method === 'trust_region_steihaug') {
        const its = num(info.cg_iters) ?? 0;
        note += ` Inner CG: ${int(its)} ${its === 1 ? 'iteration' : 'iterations'}, stopped by ${String(info.termination ?? '—').replace('_', ' ')}.`;
        extra.push({ tex: '\\text{CG its}', value: show(info.cg_iters) });
      } else if (i.method === 'trust_region_exact') {
        note += ` $\\lambda = ${tn(info.lambda)}$ solves the secular equation${info.hard_case === true ? ' (the hard case)' : ''}.`;
        extra.push({ tex: '\\lambda', value: show(info.lambda) });
      } else {
        extra.push({ tex: '\\tau', value: show(info.tau) });
        if (typeof info.note === 'string') note += ` ${info.note}`;
      }
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: '\\Delta_k', value: show(info.radius) },
          { tex: '\\rho_k', value: show(info.rho) },
          ...extra,
        ],
      };
    }
    case 'nm': {
      const op = String(info.operation ?? '');
      const trials = Array.isArray(info.trials) ? (info.trials as Record<string, unknown>[]) : [];
      const sf = Array.isArray(info.simplex_f) ? (info.simplex_f as number[]) : [];
      const co = (info.coefficients ?? {}) as Record<string, number>;
      const lines = [
        `\\bar{\\mathbf{x}} &= ${tv(info.centroid)},\\quad \\mathbf{x}_{\\text{worst}} = ${tv(info.worst)}`,
        ...trials.map(
          (t) => `\\mathbf{x}_{${OPS[String(t.op)] ?? 'r'}} &= ${tv(t.x)},\\quad f = ${tn(t.f)}`,
        ),
      ];
      const NOTE: Record<string, string> = {
        reflect: `The reflection beat the second-worst vertex but not the best: it replaces the worst vertex ($\\rho = ${tn(co.rho)}$).`,
        expand: `The reflection beat the best vertex, so the simplex expanded further ($\\chi = ${tn(co.chi)}$).`,
        contract_outside: `The reflection beat only the worst vertex: an outside contraction ($\\gamma = ${tn(co.gamma)}$).`,
        contract_inside: `The reflection was worse than every vertex: an inside contraction ($\\gamma = ${tn(co.gamma)}$).`,
        shrink: `The contraction failed, so every vertex moved toward the best one ($\\sigma = ${tn(co.sigma)}$).`,
      };
      return {
        tex: al(lines),
        note: NOTE[op] ?? op,
        quantities: [
          ...q,
          { tex: '\\text{size}', value: show(info.size) },
          { tex: 'f_{n+1} - f_1', value: show(info.f_spread) },
          { tex: 'f(\\text{vertices})', value: show(sf), wide: true },
        ],
      };
    }
    case 'powell': {
      const ls = Array.isArray(info.lines) ? (info.lines as Record<string, unknown>[]) : [];
      const lines = ls.map(
        (l, j) =>
          `\\alpha_{${j + 1}} &= ${tn(l.alpha)},\\quad ${X(`(${j + 1})`)} = ${tv(l.point)},\\ f = ${tn(l.f)}`,
      );
      const rep = num(info.replaced);
      return {
        tex: lines.length ? al(lines) : null,
        note:
          rep !== null
            ? `The extrapolation passed Powell's test ($t = ${tn(info.replace_test, 3)} < 0$): the direction of largest decrease, $\\mathbf{u}_{${rep + 1}}$, is replaced by $\\mathbf{x}_n - \\mathbf{x}_0$.`
            : 'The direction set is kept: Powell’s replacement test did not pass.',
        quantities: [
          ...q,
          { tex: '\\Delta f_{\\max}', value: show(info.largest_decrease) },
          { tex: '\\text{replaced}', value: rep === null ? '—' : `u${rep + 1}` },
        ],
      };
    }
    case 'hj': {
      const OUT: Record<string, string> = {
        explore_success:
          'The exploratory moves around the base point found a lower value: a new base point.',
        step_reduced: `No probe decreased $f$, so the step shrinks: $h = ${tn(info.step)} \\to ${tn(info.new_step)}$.`,
        pattern_success:
          'The pattern move, continued by exploration, found a lower value: the base moves on.',
        pattern_failed: 'The pattern move failed; exploration returns to the base point.',
      };
      const probes = Array.isArray(info.probes) ? info.probes.length : 0;
      return {
        tex: al([
          `\\mathbf{p} &= \\mathbf{b} + (\\mathbf{b} - \\mathbf{b}_{\\text{old}}) = ${tv(info.pattern_point)}`,
          `\\mathbf{b} &= ${tv(info.base)},\\quad h = ${tn(info.step)}`,
        ]),
        note: (OUT[String(info.outcome)] ?? String(info.outcome)) + ` ${int(probes)} probes.`,
        quantities: [
          ...q,
          { tex: 'h', value: show(info.step) },
          { tex: '\\text{move}', value: String(info.move ?? '—') },
        ],
      };
    }
    case 'compass': {
      const polls = Array.isArray(info.polls) ? (info.polls as Record<string, unknown>[]) : [];
      const DIRS = ['+\\mathbf{e}_1', '+\\mathbf{e}_2', '-\\mathbf{e}_1', '-\\mathbf{e}_2'];
      const d = num(info.direction);
      return {
        tex: al([
          `${X(k)} &= ${X(k - 1)} + \\Delta_{${k}}\\,\\mathbf{d},\\quad \\Delta_{${k}} = ${tn(info.step)}`,
          ...polls.map(
            (p, j) =>
              `f(${X(k - 1)} ${['+', '+', '-', '-'][j]} \\Delta\\,\\mathbf{e}_{${(j % 2) + 1}}) &= ${tn(p.f)}`,
          ),
        ]),
        note:
          info.success === true
            ? `The poll along $${d === null ? '\\text{—}' : DIRS[d]}$ decreased $f$ after ${polls.length === 1 ? 'one evaluation' : `${polls.length} evaluations`}; $\\Delta$ stays.`
            : `None of the four poll points decreased $f$: $\\Delta$ shrinks to $${tn(info.new_step)}$.`,
        quantities: [...q, { tex: '\\Delta_k', value: show(info.step) }],
      };
    }

    case 'schedule': {
      const h = num(info.h);
      const a = num(info.alpha);
      const L = h !== null && a !== null && a > 0 ? h / a : null;
      const tex = al([
        `${X(k)} &= ${X(k - 1)} - \\frac{h_{${k - 1}}}{L}\\,${GF(k - 1)}`,
        `&= ${tv(prev.x)} - \\frac{${tn(h)}}{${tn(L)}}\\,${tv(prev.info.grad)}`,
        `&= ${tv(s.x)}`,
      ]);
      let note =
        h === null
          ? ''
          : h > 2
            ? `A long step: $h = ${tn(h)} > 2$, beyond the longest safe constant step. ` +
              (typeof s.fun === 'number' && typeof prev.fun === 'number' && s.fun > prev.fun
                ? `$f$ rose by $${tn(s.fun - prev.fun)}$; the short steps that follow repair it.`
                : '$f$ still fell.')
            : `A short step: $h = ${tn(h)} \\le 2$.`;
      const cert = certificateNumbers(i);
      if (num(info.bound_dist) !== null && cert) {
        const bd = num(info.bound_dist)!;
        note += ` Certificate after ${int(Math.round(k))} steps: $\\|\\mathbf{x}_k - \\mathbf{x}^\\star\\|^2 = ${tn(cert.distK, 3)} \\le \\tau^m\\|\\mathbf{x}_0 - \\mathbf{x}^\\star\\|^2 = ${tn(bd * cert.dist0, 3)}$.`;
      } else if (num(info.bound_f) !== null && cert && L !== null) {
        const bf = num(info.bound_f)!;
        note += ` Certificate at $k = 2^j - 1$: $f(\\mathbf{x}_k) - f^\\star = ${tn(cert.gap, 3)} \\le r_j L\\|\\mathbf{x}_0 - \\mathbf{x}^\\star\\|^2 = ${tn(bf * L * cert.dist0, 3)}$.`;
      } else if (info.checkpoint === true)
        note += ' The pattern ends here; the proved rate has no explicit constant to check.';
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: `h_{${k - 1}}`, value: show(h) },
          { tex: `\\alpha_{${k}} = h/L`, value: show(a) },
        ],
      };
    }
    case 'ogm': {
      const t0 = num(prev.info.theta);
      const t1 = num(info.theta);
      const w1 = t0 !== null && t1 !== null ? (t0 - 1) / t1 : null;
      const w2 = t0 !== null && t1 !== null ? t0 / t1 : null;
      const tex = al([
        `\\mathbf{y}_{${k}} &= ${X(k - 1)} - \\tfrac1L\\,${GF(k - 1)} = ${tv(info.y)}`,
        `${X(k)} &= \\mathbf{y}_{${k}} + ${tn(w1, 3)}\\,(\\mathbf{y}_{${k}} - \\mathbf{y}_{${k - 1}}) + ${tn(w2, 3)}\\,(\\mathbf{y}_{${k}} - ${X(k - 1)})`,
        `&= ${tv(s.x)}`,
      ]);
      let note = `$\\theta_{${k}} = ${tn(t1)}$: the momentum weights are $(\\theta_{${k - 1}} - 1)/\\theta_{${k}} = ${tn(w1, 3)}$ and $\\theta_{${k - 1}}/\\theta_{${k}} = ${tn(w2, 3)}$.`;
      const cert = certificateNumbers(i);
      const a = num(info.alpha);
      if (num(info.bound_f) !== null && cert && a !== null && a > 0)
        note += ` At $k = N$ the certificate holds: $f(\\mathbf{x}_N) - f^\\star = ${tn(cert.gap, 3)} \\le L\\|\\mathbf{x}_0 - \\mathbf{x}^\\star\\|^2/(2\\theta_N^2) = ${tn((num(info.bound_f)! / a) * cert.dist0, 3)}$.`;
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: `\\theta_{${k}}`, value: show(t1) },
          { tex: '\\mathbf{y}_k', value: show(info.y), wide: true },
        ],
      };
    }
    case 'fista': {
      const beta = num(info.beta);
      const L = num(info.L);
      const tex = al([
        `\\mathbf{y}_{${k}} &= ${X(k - 1)} + \\beta_{${k}}\\,(${X(k - 1)} - ${X(Math.max(0, k - 2))}) = ${tv(info.y)}`,
        `${X(k)} &= \\mathbf{y}_{${k}} - \\tfrac{1}{L_{${k}}}\\,\\nabla f(\\mathbf{y}_{${k}}) = ${tv(s.x)},\\quad L_{${k}} = ${tn(L)}`,
      ]);
      let note =
        beta === null || beta === 0
          ? 'No momentum in this step ($\\beta = 0$: the first step, or the first after a restart).'
          : `The momentum coefficient is $\\beta_{${k}} = ${tn(beta, 3)}$; it grows toward 1 until a restart.`;
      const trials = Array.isArray(info.trials) ? info.trials.length : 0;
      if (trials > 0)
        note += ` Backtracking tried ${plural(trials, 'value')} of $L$ and accepted $L_{${k}} = ${tn(L)}$.`;
      else note += ` A constant step $1/L = ${tn(num(info.alpha))}$.`;
      if (info.restarted === true)
        note += ` Restart: ${RESTART_TEST[String(P.restart ?? 'gradient')] ?? 'the test fired'}, so $t$ resets to 1 and the next step has no momentum.`;
      return {
        tex,
        note,
        quantities: [
          { tex: 'k', value: int(s.k) },
          { tex: `f(${X('k')})`, value: show(s.fun) },
          { tex: '\\|\\nabla f(\\mathbf{y}_k)\\|_2', value: show(info.grad_norm_y) },
          { tex: X('k'), value: show(s.x), wide: true },
          { tex: `\\beta_{${k}}`, value: show(beta) },
          { tex: `L_{${k}}`, value: show(L) },
          { tex: 't_{k+1}', value: show(info.t) },
        ],
      };
    }
    case 'anderson': {
      const c = Array.isArray(info.coefficients) ? (info.coefficients as number[]) : [];
      const mk = num(info.memory) ?? 0;
      const lines =
        mk === 0
          ? [`${X(k)} &= ${X(k - 1)} - \\beta\\alpha\\,${GF(k - 1)} = ${tv(s.x)}`]
          : [
              `\\mathbf{c}^\\star &= ${tv(c, 3)}`,
              `\\bar{\\mathbf{x}} &= \\textstyle\\sum_i c_i^\\star\\,\\mathbf{x}_i = ${tv(info.x_bar)}`,
              `${X(k)} &= \\bar{\\mathbf{x}} - \\beta\\alpha \\textstyle\\sum_i c_i^\\star \\nabla f(\\mathbf{x}_i) = ${tv(s.x)}`,
            ];
      const fNewest =
        typeof prev.gradNorm === 'number' && num(info.alpha) !== null
          ? num(info.alpha)! * prev.gradNorm
          : null;
      const res = num(info.lsq_residual);
      let note =
        mk === 0
          ? 'A plain gradient step: there is no history to mix yet (or $m = 0$).'
          : `${int(mk + 1)} iterates mixed. The mixed residual $\\|F\\mathbf{c}^\\star\\| = ${tn(res, 3)}$` +
            (fNewest !== null && res !== null && fNewest > 0
              ? ` is $${tn(res / fNewest, 3)}$ times the newest one, $\\alpha\\|\\nabla f(${X(k - 1)})\\|$.`
              : '.');
      if (mk > 0 && c.some((w) => w < 0))
        note += ' Negative weights extrapolate beyond the history.';
      const eig = info.hess_eigs;
      if (Array.isArray(eig) && typeof eig[0] === 'number')
        note +=
          (eig[0] as number) < 0
            ? ` The gradient test passed, but $\\nabla^2 f$ has the eigenvalues $${tv(eig, 3)}$: not a minimizer.`
            : ` The gradient test passed and $\\nabla^2 f$ has the eigenvalues $${tv(eig, 3)}$.`;
      return {
        tex: al(lines),
        note,
        quantities: [
          ...q,
          { tex: 'm_k', value: show(info.memory) },
          { tex: '\\|F\\mathbf{c}^\\star\\|', value: show(res) },
          { tex: '\\operatorname{cond}', value: show(info.cond) },
        ],
      };
    }
    case 'arc': {
      const accepted = info.accepted === true;
      const tex = al([
        `\\rho_{${k}} &= \\frac{f(${X(k - 1)}) - f(${X(k - 1)} + \\mathbf{s})}{f(${X(k - 1)}) - m(\\mathbf{s})} = \\frac{${tn(info.actual)}}{${tn(info.predicted)}} = ${tn(info.rho, 3)}`,
        `\\sigma &: ${tn(info.sigma)} \\to ${tn(info.new_sigma)},\\quad \\|\\mathbf{s}\\| = \\lambda/\\sigma = ${tn(info.step_norm)},\\quad ${X(k)} = ${accepted ? `${X(k - 1)} + \\mathbf{s}` : X(k - 1)}`,
      ]);
      let note = `This step is ${ITERATION_CLASS[String(info.iteration)] ?? String(info.iteration)}.`;
      const lm = num(info.lambda_min);
      note += ` $\\lambda = \\sigma\\|\\mathbf{s}\\| = ${tn(info.lambda)}$${info.hard_case === true ? ' (the hard case)' : ''} makes $\\nabla^2 f + \\lambda I$ positive semidefinite`;
      note +=
        lm !== null && lm < 0
          ? `; $\\lambda_{\\min}(\\nabla^2 f) = ${tn(lm, 3)} < 0$, so the step follows negative curvature.`
          : '.';
      if (typeof info.note === 'string') note += ` ${info.note}`;
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: '\\sigma_k', value: show(info.sigma) },
          { tex: '\\rho_k', value: show(info.rho) },
          { tex: '\\lambda', value: show(info.lambda) },
        ],
      };
    }
    case 'regnewton': {
      const lam = num(info.lambda);
      const tex = al([
        `&\\left(${tm(prev.info.hess, 3)} + ${tn(lam, 3)}\\,I\\right)\\mathbf{s} = -${tc(prev.info.grad)}`,
        `&\\mathbf{s} = ${tv(info.direction)},\\quad ${X(k)} = ${X(k - 1)} + \\mathbf{s} = ${tv(s.x)}`,
      ]);
      const n = num(info.inner_iters) ?? 1;
      const variant = String(P.variant ?? 'adan');
      let note =
        variant === 'fixed'
          ? `$\\lambda = \\sqrt{H\\|\\nabla f\\|} = ${tn(lam)}$ with the fixed $H = ${tn(info.H_reg)}$.`
          : variant === 'super_universal'
            ? `$\\lambda = H\\|\\nabla f\\|^{\\alpha} = ${tn(lam)}$ with $H = ${tn(info.H_reg)}$ after ${plural(n, 'trial')} of the test $\\langle\\nabla f(\\mathbf{x}_+), \\mathbf{x} - \\mathbf{x}_+\\rangle \\ge \\|\\nabla f(\\mathbf{x}_+)\\|^2/(4\\lambda)$.`
            : `AdaN starts each step from $H_{k-1}/4$ and doubles it until both decrease tests pass: ${n === 1 ? 'one doubling' : `${int(n)} doublings`}, $H_{${k}} = ${tn(info.H_reg)}$, $\\lambda = \\sqrt{H\\|\\nabla f\\|} = ${tn(lam)}$.`;
      if (info.pd === false)
        note +=
          ' $\\nabla^2 f + \\lambda I$ is indefinite here: the step is taken as written (the theory needs a convex $f$).';
      if (info.descent === false)
        note += ' The step goes uphill ($\\nabla f^{\\top}\\mathbf{s} > 0$).';
      return {
        tex,
        note,
        quantities: [
          ...q,
          { tex: '\\lambda_k', value: show(lam) },
          { tex: 'H_k', value: show(info.H_reg) },
          { tex: '\\text{trials}', value: int(n) },
        ],
      };
    }
  }
}

const OPS: Record<string, string> = {
  reflect: 'r',
  expand: 'e',
  contract_outside: 'oc',
  contract_inside: 'ic',
};

const SUB = '₀₁₂₃₄₅₆₇₈₉';
/** k as Unicode subscript digits (for plain-text notes: 𝐱₁₂). */
export function sub(k: number): string {
  return String(k)
    .split('')
    .map((c) => (c === '-' ? '₋' : SUB[Number(c)]))
    .join('');
}

/** At most this many steps of a long run are read to pick the layout samples. */
const SAMPLE_READS = 600;

/**
 * Steps whose "This step" text is the largest of its kind in the run: for each shape (number of
 * formula rows and of quantities), the step with the longest formula and note. Typeset hidden
 * in the same grid cell as the current step, they hold the block at the height of the tallest
 * step and give the run one formula size (FitGroup), so the panel does not move while it plays.
 * A long run is read at an even stride (plus its first and last steps).
 */
export function layoutSamples(i: Omit<StepInput, 'g'>): number[] {
  const n = i.trace.length;
  const stride = Math.max(1, Math.ceil((n - 1) / SAMPLE_READS));
  const best = new Map<string, { g: number; score: number }>();
  const read = (g: number) => {
    // The strip has one fixed height: the step text alone decides the shape.
    const v = stepViewRaw({ ...i, g });
    if (!v.tex) return;
    const rows = (v.tex.match(/\\\\/g) ?? []).length;
    const key = `${rows}|${v.quantities.length}`;
    const score =
      v.tex.length + 3 * v.note.length + 2 * v.quantities.reduce((a, q) => a + q.value.length, 0);
    const seen = best.get(key);
    if (!seen || score > seen.score) best.set(key, { g, score });
  };
  for (let g = 1; g < n; g += stride) read(g);
  if (n > 2 && (n - 2) % stride !== 0) read(n - 1);
  return [0, ...[...best.values()].map((b) => b.g)];
}
