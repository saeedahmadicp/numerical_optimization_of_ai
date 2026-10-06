/**
 * The update rule of the MethodCard with this step's numbers filled in: the general rule on the
 * first line, the same rule with h, N and the sampled values on the second, and the result on a
 * third line (so the filled line fits a phone). One symbol per estimate everywhere in the lab:
 * `estimateTex` (generic, for headers) and `estimateSymbol` (with this step's N, m or k).
 */
import type { Step } from '../../core/types';
import { texNum } from './quadFormat';

type XY = [number, number];

const nodesOf = (s: Step) => (Array.isArray(s.info.nodes) ? (s.info.nodes as XY[]) : null);

/** `c_0 f_0 + c_1 f_1 + … + c_N f_N` with the first `head` and the last term written out. */
function weightedSum(values: readonly number[], coeffs: (i: number) => string, head = 1): string {
  const n = values.length;
  const term = (i: number) => {
    const c = coeffs(i);
    const v = texNum(values[i], 3);
    return c === '' ? v : `${c}(${v})`;
  };
  if (n <= head + 2) return values.map((_, i) => term(i)).join(' + ');
  const parts: string[] = [];
  for (let i = 0; i < head; i++) parts.push(term(i));
  parts.push('\\cdots', term(n - 1));
  return parts.join(' + ');
}

const SYMBOL: Record<string, string> = {
  left_riemann: 'L',
  right_riemann: 'R',
  midpoint_rule: 'M',
  trapezoid: 'T',
  simpson: 'S',
  simpson_38: 'S^{3/8}',
  boole: 'B',
};

/** Generic symbol of a method's estimate: `T_N`, `S_N`, `R_{k,k}`, `G_m`, `I_k`, `I_N`, … */
export function estimateTex(method: string): string {
  if (SYMBOL[method]) return SYMBOL[method].includes('^') ? 'S_N^{3/8}' : `${SYMBOL[method]}_N`;
  if (method === 'romberg') return 'R_{k,k}';
  if (method === 'gauss_legendre') return 'G_m';
  if (method === 'clenshaw_curtis') return 'C_n';
  if (method === 'gauss_patterson') return 'P_N';
  if (method === 'monte_carlo_integration') return 'I_N';
  return 'I_k';
}

/**
 * Symbol of the estimate at this step, e.g. `T_{64}`, `R_{3,3}`, `G_{7}`, `C_{16}` (degree n,
 * n + 1 nodes), `P_{15}` (15 nodes), `I_{3200}`, `I_{12}`.
 */
export function estimateSymbol(method: string, step: Step): string {
  const info = step.info;
  if (SYMBOL[method]) {
    const s = SYMBOL[method];
    const N = info.n_panels as number;
    return s.includes('^') ? `S_{${N}}^{3/8}` : `${s}_{${N}}`;
  }
  if (method === 'romberg') return `R_{${step.k},${step.k}}`;
  if (method === 'gauss_legendre') return `G_{${info.n_points as number}}`;
  if (method === 'clenshaw_curtis') return `C_{${info.n as number}}`;
  if (method === 'gauss_patterson') return `P_{${info.n_points as number}}`;
  if (method === 'monte_carlo_integration') return `I_{${info.n_samples as number}}`;
  return `I_{${step.k}}`;
}

/** Composite coefficient pattern of node i (of n) for the bracket sum. */
function compositeCoeff(method: string, i: number, n: number): string {
  const last = i === n - 1;
  switch (method) {
    case 'trapezoid':
      return i === 0 || last ? '\\tfrac12' : '';
    case 'simpson':
      return i === 0 || last ? '' : i % 2 ? '4' : '2';
    case 'simpson_38':
      return i === 0 || last ? '' : i % 3 ? '3' : '2';
    case 'boole':
      if (i === 0 || last) return '7';
      return ['14', '32', '12', '32'][i % 4];
    default:
      return '';
  }
}

/** Leading terms written out per rule: one full period of the weight pattern plus f₀. */
const SPAN_HEAD: Record<string, number> = { trapezoid: 2, simpson: 3, simpson_38: 4, boole: 5 };

const PREFIX: Record<string, (h: string) => string> = {
  left_riemann: (h) => h,
  right_riemann: (h) => h,
  midpoint_rule: (h) => h,
  trapezoid: (h) => h,
  simpson: (h) => `\\tfrac{${h}}{3}`,
  simpson_38: (h) => `\\tfrac{3${h}}{8}`,
  boole: (h) => `\\tfrac{2${h}}{45}`,
};

/**
 * Two-line `aligned` TeX: the general `rule`, then the instantiated rule at `step`.
 * `prev` is the previous step (Romberg needs R(k−1, k−1)); `ab` is the interval.
 */
export function filledRule(
  method: string,
  rule: string,
  step: Step | undefined,
  prev: Step | undefined,
  ab: readonly [number, number],
): string {
  if (!step) return rule;
  const est = texNum(step.info.estimate as number, 7);
  const sym = estimateSymbol(method, step);
  let line: string | null = null;
  /** The result, set on its own third line when the filled line is long. */
  let result: string | null = null;
  if (PREFIX[method]) {
    const nodes = nodesOf(step);
    const h = texNum(step.stepSize, 4);
    if (nodes) {
      const vals = nodes.map((p) => p[1]);
      // Newton–Cotes rules: enough leading terms to show the weight pattern (1, 4, 2, 4, …).
      const head = SPAN_HEAD[method] ?? 1;
      const sum = weightedSum(vals, (i) => compositeCoeff(method, i, vals.length), head);
      // h stays a symbol in the bracket line (it must fit a phone); its value joins the result.
      line = `${sym} = ${PREFIX[method]('h')}\\left[${sum}\\right]`;
      result = `${est},\\qquad h = ${h}`;
    } else line = `${sym} = ${est}\\quad (h = ${h})`;
  } else if (method === 'romberg') {
    const row = step.info.row as number[];
    const k = step.k;
    if (k === 0) line = `R_{0,0} = \\tfrac{b-a}{2}\\,[f(a) + f(b)] = ${est}`;
    else {
      const a = texNum(row[k - 1], 8);
      const b = texNum((prev?.info.row as number[] | undefined)?.[k - 1] ?? NaN, 8);
      line = `R_{${k},${k}} = ${a} + \\frac{${a} - ${b}}{${4 ** k - 1}}`;
      result = est;
    }
  } else if (method === 'gauss_legendre') {
    const nodes = nodesOf(step) ?? [];
    const w = (step.info.weights as number[] | null) ?? [];
    // The trace's weights are already scaled to [a, b]: w̃ᵢ = (b − a)/2 · wᵢ, at xᵢ (not tᵢ).
    const terms = nodes.map((p, i) => `${texNum(w[i], 3)}\\,f(${texNum(p[0], 3)})`);
    const shown =
      terms.length <= 3 ? terms.join(' + ') : `${terms[0]} + \\cdots + ${terms[terms.length - 1]}`;
    line = `${sym} = \\textstyle\\sum_i \\tilde w_i\\, f(x_i) = ${shown}`;
    result = `${est},\\qquad \\tilde w_i = \\tfrac{b-a}{2}\\,w_i`;
  } else if (method === 'clenshaw_curtis' || method === 'gauss_patterson') {
    const nodes = nodesOf(step);
    const w = (step.info.weights as number[] | null) ?? [];
    const cc = method === 'clenshaw_curtis';
    const j = cc ? 'j' : 'i';
    if (nodes) {
      // The trace lists the nodes from b down to a; the sum reads from a to b.
      const terms = nodes.map((p, i) => `${texNum(w[i], 3)}\\,f(${texNum(p[0], 3)})`).reverse();
      const shown =
        terms.length <= 3
          ? terms.join(' + ')
          : `${terms[0]} + \\cdots + ${terms[terms.length - 1]}`;
      line = `${sym} = \\textstyle\\sum_${j} \\tilde w_${j}\\, f(x_${j}) = ${shown}`;
    } else line = `${sym} = \\textstyle\\sum_${j} \\tilde w_${j}\\, f(x_${j})`;
    const fresh = step.info.new_nodes as number;
    const N = step.info.n_points as number;
    const reuse = step.k > 0 ? `,\\qquad \\text{${N - fresh} of ${N} values reused}` : '';
    result = `${est}${reuse}`;
  } else if (method === 'adaptive_simpson') {
    const iv = step.info.interval as XY | null;
    if (!iv) line = `S_1[a,b] = \\tfrac{b-a}{6}\\left[f(a) + 4f(m) + f(b)\\right] = ${est}`;
    else {
      const d = texNum(15 * (step.info.err_est as number), 3);
      const tau = texNum(15 * (step.info.tol_local as number), 3);
      const ok = step.info.passed ? '\\le' : '>';
      line = `|S_2 - S_1| = ${d} ${ok} 15\\tau = ${tau}`;
    }
  } else if (method === 'monte_carlo_integration') {
    const N = step.info.n_samples as number;
    const se = step.info.err_est as number | null;
    const width = texNum(ab[1] - ab[0], 5);
    const pm = se === null || se === undefined ? '' : `\\pm ${texNum(se, 2)}`;
    line = `I_{${N}} = \\tfrac{${width}}{${N}}\\textstyle\\sum_i f(U_i) = ${est} ${pm}`;
  }
  if (!line) return rule;
  const third = result === null ? '' : `\\\\[2pt]&\\phantom{${sym}} = ${result}`;
  return `\\begin{aligned}&${rule}\\\\[2pt]&${line}${third}\\end{aligned}`;
}
