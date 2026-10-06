/**
 * The MethodCard's update rule with the current step's numbers filled in: the textbook rule on
 * the first line, the same rule evaluated with this step's values on the second
 * (`x₃ = 2.1 − (0.061)/(11.23) = 2.0946`). Built from `Step.info` only, so it shows exactly
 * the quantities the method used.
 */
import type { Step } from '../../core/types';

/**
 * A number in TeX: `digits` significant digits, ×10ⁿ outside [10⁻³, 10⁵) (4 digits there,
 * unless more than 6 are asked for).
 */
export function tex(v: unknown, digits = 6): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential((digits > 6 ? digits : Math.min(4, digits)) - 1).split('e');
    return `${m.replace(/\.?0+$/, '')} \\times 10^{${Number(e)}}`;
  }
  return String(Number(v.toPrecision(digits)));
}

/**
 * Significant digits that tell the points of an interval apart: two more than the number of
 * leading digits its ends share, ⌈log₁₀(max(|lo|, |hi|)/(hi − lo))⌉ + 2, in [4, 15]. With 4
 * digits for every operand, bisection on [2.0945514814, 2.0945514816] read
 * "(2.095 + 2.095)/2 = 2.09455".
 */
export function digitsFor(lo: number, hi: number): number {
  const w = Math.abs(hi - lo);
  const m = Math.max(Math.abs(lo), Math.abs(hi));
  if (!Number.isFinite(w) || !Number.isFinite(m)) return 6;
  if (m === 0) return 4;
  if (w === 0) return 15;
  return Math.max(4, Math.min(15, Math.ceil(Math.log10(m / w)) + 2));
}

/** An operand inside the filled rule: 4 digits (or `d`), in parentheses when negative. */
const p = (v: unknown, d = 4) => (typeof v === 'number' && v < 0 ? `(${tex(v, d)})` : tex(v, d));

/** An operand squared: parenthesized when negative or in ×10ⁿ form (no double superscript). */
const sq = (v: unknown) => {
  const t = tex(v, 4);
  return typeof v === 'number' && (v < 0 || t.includes('^')) ? `\\left(${t}\\right)^2` : `${t}^2`;
};

const num = (v: unknown): number => (typeof v === 'number' ? v : NaN);
const pair = (v: unknown): [number, number] =>
  Array.isArray(v) ? [num(v[0]), num(v[1])] : [NaN, NaN];

/** The rule, its evaluation, and further aligned lines (the kept bracket, …). */
function lines(rule: string, value: string, ...more: string[]): string {
  const rest = more
    .filter(Boolean)
    .map((l) => ` \\\\ ${l}`)
    .join('');
  return `\\begin{aligned} ${rule} \\\\ &= ${value}${rest} \\end{aligned}`;
}

/** The filled rule for step `s` of `method`, or null to keep the static rule. */
export function filledRule(method: string, s: Step | undefined): string | null {
  if (!s) return null;
  const k = s.k;
  const xk = `x_{${k}}`;
  const xp = `x_{${k - 1}}`;
  const x = num(s.x);
  const info = s.info;
  const [lo, hi] = pair(info.bracket);
  const [nlo, nhi] = pair(info.new_bracket);
  // Digits that separate the ends of the bracket the step used (operands, result) and of the
  // bracket it keeps.
  const d = Number.isFinite(lo) ? digitsFor(lo, hi) : 6;
  const dn = Number.isFinite(nlo) ? digitsFor(nlo, nhi) : d;
  // The kept bracket, its ends named where they are this step's points: [a_k, x_k] says which
  // half survived, and stays short when the ends need 13 digits. Brent's rule already uses b
  // for its best point, so Brent's ends stay numeric.
  const named = method !== 'brent';
  const end = (v: number) =>
    v === x
      ? xk
      : named && v === lo
        ? `a_{${k}}`
        : named && v === hi
          ? `b_{${k}}`
          : named && method === 'ridders' && v === info.midpoint
            ? 'm'
            : tex(v, dn);
  const keep = Number.isFinite(nlo)
    ? `&\\Rightarrow\\; [a_{${k + 1}},\\, b_{${k + 1}}] = [${end(nlo)},\\ ${end(nhi)}]`
    : '';
  const xd = tex(x, Math.max(d, dn));
  // Brent, Chandrupatla, ITP: the estimate is the kept end with the smaller |f|.
  const best =
    typeof info.best === 'number'
      ? `&\\hat x_{${k}} = ${tex(info.best, Math.max(d, dn))} \\quad (\\text{end with smaller } |f|)`
      : '';
  // Open methods: digits that separate x_{k−1} from x_k (the step is tiny late in a run).
  const od = (prev: unknown) => (typeof prev === 'number' ? digitsFor(prev, x) : 6);

  switch (method) {
    case 'bisection':
      return lines(
        `${xk} &= \\dfrac{a_{${k}} + b_{${k}}}{2}`,
        `\\dfrac{${p(lo, d)} + ${p(hi, d)}}{2}`,
        `&= ${xd}`,
        keep,
      );
    case 'regula_falsi':
    case 'illinois':
    case 'pegasus':
    case 'anderson_bjorck': {
      if (info.step === 'verify') return lines(`${xk} &= ${xp} \\pm \\mathrm{tol}`, xd, keep);
      if (info.step === 'bisection') return lines(`${xk} &= \\tfrac{a + b}{2}`, xd, keep);
      const ch = info.chord as number[][] | null;
      if (!ch) return null;
      const [[a, fa], [b, fb]] = ch;
      const scale = num(info.scale);
      // `scale` is applied AFTER this step, to the retained end's ordinate in the next chord.
      const next =
        method !== 'regula_falsi' && Number.isFinite(scale) && scale !== 1
          ? `,\\quad \\text{next } f_{\\text{kept}} \\leftarrow ${tex(scale, 4)}\\, f_{\\text{kept}}`
          : '';
      return lines(
        `${xk} &= b - f_b\\,\\dfrac{b - a}{f_b - f_a}`,
        `${p(b, d)} - ${p(fb)}\\,\\dfrac{${p(b, d)} - ${p(a, d)}}{${p(fb)} - ${p(fa)}} = ${xd}`,
        keep ? `${keep}${next}` : next ? `&${next.slice(1)}` : '',
      );
    }
    case 'ridders': {
      if (info.step === 'verify') return lines(`${xk} &= ${xp} \\pm \\mathrm{tol}`, xd, keep);
      const m = num(info.midpoint),
        fm = num(info.f_mid);
      const tr = info.transformed as number[][] | null;
      const q = num(info.exp_factor);
      if (!tr) return lines(`${xk} &= m`, `${p(m, d)}`, keep);
      const fa = tr[0][1],
        fb = tr[2][1] / (q * q);
      return lines(
        `${xk} &= m + (m - a)\\dfrac{\\operatorname{sign}(f_a - f_b)\\, f_m}{\\sqrt{f_m^2 - f_a f_b}}`,
        `${p(m, d)} + ${p(m - lo)}\\dfrac{${fa - fb < 0 ? '-' : ''}${p(fm)}}{\\sqrt{${sq(fm)} - ${p(fa)} \\cdot ${p(fb)}}} = ${xd}`,
        keep,
      );
    }
    case 'brent': {
      const kind = info.step as string;
      const att = info.attempted as string | null;
      const word =
        kind === 'bisection'
          ? att
            ? `\\text{bisection (${att === 'secant' ? 'secant' : 'IQI'} rejected)}`
            : '\\text{bisection}'
          : kind === 'secant'
            ? '\\text{secant}'
            : '\\text{inverse quadratic}';
      return lines(
        `${xk} &= b + d, \\quad d = ${word}`,
        `${xd}, \\quad \\mathrm{tol} = ${tex(info.tol, 3)}`,
        keep,
        best,
      );
    }
    case 'chandrupatla': {
      const t = num(info.t);
      const word = info.step === 'bisection' ? '\\text{bisection}' : '\\text{inverse quadratic}';
      return lines(
        `${xk} &= x_1 + t\\,(x_2 - x_1)`,
        xd,
        `&t = ${tex(t, 4)} \\quad (${word})`,
        keep,
        best,
      );
    }
    case 'itp': {
      // σ = sign(x½ − x_f): the truncation moves x_f toward the midpoint, the projection
      // moves the midpoint back toward x_f. Both signs are known for the step.
      const xh = num(info.x_half),
        xf = num(info.x_f);
      const sigma = xh > xf ? 1 : xh < xf ? -1 : 0;
      const proj = info.projected === true;
      const truncated = !proj && num(info.delta) <= Math.abs(xh - xf);
      const op = (sg: number) => (sg >= 0 ? '+' : '-');
      const rule = proj
        ? `${xk} &= x_{1/2} - \\sigma r_k, \\quad \\sigma = ${sigma < 0 ? '-1' : '+1'}`
        : truncated
          ? `${xk} &= x_f + \\sigma \\delta, \\quad \\sigma = ${sigma < 0 ? '-1' : '+1'}`
          : `${xk} &= x_{1/2} \\quad (\\delta > |x_{1/2} - x_f|)`;
      const value = proj
        ? `${tex(xh, d)} ${op(-sigma)} ${tex(num(info.r), 3)} = ${xd}`
        : truncated
          ? `${tex(xf, d)} ${op(sigma)} ${tex(num(info.delta), 3)} = ${xd}`
          : xd;
      return lines(rule, value, keep, best);
    }
    case 'newton': {
      const tg = info.tangent as { point: number[]; slope: number } | undefined;
      if (!tg) return `x_0 = ${tex(x)}`;
      return lines(
        `${xk} &= ${xp} - \\dfrac{f(${xp})}{f'(${xp})}`,
        `${p(tg.point[0], od(tg.point[0]))} - \\dfrac{${p(tg.point[1])}}{${p(tg.slope)}} = ${tex(x, od(tg.point[0]))}`,
      );
    }
    case 'secant': {
      const ch = info.chord as number[][] | undefined;
      if (!ch) return `x_0 = ${tex(x)}`;
      const [[x2, f2], [x1, f1]] = ch;
      return lines(
        `${xk} &= ${xp} - f_{${k - 1}}\\,\\dfrac{${xp} - x_{${k - 2}}}{f_{${k - 1}} - f_{${k - 2}}}`,
        `${p(x1, od(x1))} - ${p(f1)}\\,\\dfrac{${p(x1, od(x1))} - ${p(x2, od(x1))}}{${p(f1)} - ${p(f2)}} = ${tex(x, od(x1))}`,
      );
    }
    case 'halley': {
      const der = info.derivatives as number[] | undefined;
      if (!der) return `x_0 = ${tex(x)}`;
      const [f0, d1, d2] = der;
      return lines(
        `${xk} &= ${xp} - \\dfrac{2 f f'}{2 f'^2 - f f''}`,
        `${p(info.previous, od(info.previous))} - \\dfrac{2 \\cdot ${p(f0)} \\cdot ${p(d1)}}{2 \\cdot ${sq(d1)} - ${p(f0)} \\cdot ${p(d2)}} = ${tex(x, od(info.previous))}`,
      );
    }
    case 'steffensen': {
      const ch = info.chord as number[][] | undefined;
      if (!ch) return `x_0 = ${tex(x)}`;
      const [[x0, f0], [, fz]] = ch;
      return lines(
        `${xk} &= ${xp} - \\dfrac{f(${xp})^2}{f(${xp} + f(${xp})) - f(${xp})}`,
        `${p(x0, od(x0))} - \\dfrac{${sq(f0)}}{${p(fz)} - ${p(f0)}} = ${tex(x, od(x0))}`,
      );
    }
    case 'muller': {
      const par = info.parabola as { a: number; b: number; c: number } | undefined;
      if (!par) return `x_0 = ${tex(x)}`;
      return lines(
        `${xk} &= ${xp} - \\dfrac{2c}{b \\pm \\sqrt{b^2 - 4ac}}`,
        `a = ${tex(par.a)},\\ b = ${tex(par.b)},\\ c = ${tex(par.c)} \\;\\Rightarrow\\; ${xk} = ${tex(x)}`,
      );
    }
    case 'inverse_quadratic_interpolation': {
      const ip = info.inverse_parabola as { a: number; b: number; c: number } | undefined;
      if (!ip) return `x_0 = ${tex(x)}`;
      return lines(
        `x(y) &= a y^2 + b y + c, \\quad ${xk} = x(0) = c`,
        `${tex(ip.a)}\\,y^2 ${ip.b < 0 ? '-' : '+'} ${tex(Math.abs(ip.b))}\\,y ${ip.c < 0 ? '-' : '+'} ${tex(Math.abs(ip.c))} \\;\\Rightarrow\\; ${xk} = ${tex(x)}`,
      );
    }
    case 'fixed_point': {
      const prev = info.previous;
      if (typeof prev !== 'number') return `x_0 = ${tex(x)}`;
      const lam = num(info.lam);
      const cob = info.cobweb as number[][];
      const fprev = (prev - cob[1][1]) / lam;
      return lines(
        `${xk} &= ${xp} - \\lambda f(${xp})`,
        `${p(prev, od(prev))} - ${p(lam)} \\cdot ${p(fprev)} = ${tex(x, od(prev))}`,
      );
    }
    default:
      return null;
  }
}

// ── The card's live quantities ──────────────────────────────────────────────────────────

export interface CardQuantity {
  tex: string;
  key: string;
}

const STEP_Q: CardQuantity = { tex: '|x_k - x_{k-1}|', key: 'stepSize' };
const WIDTH_Q: CardQuantity = { tex: 'b_{k+1} - a_{k+1}', key: 'info.width' };

/** What the MethodCard shows under the rule, per method (keys of `cardStep`'s step). */
export const CARD_QUANTITIES: Record<string, CardQuantity[]> = {
  bisection: [
    { tex: '(b_k - a_k)/2', key: 'stepSize' },
    { tex: 'f(a_k)', key: 'info.fa' },
    { tex: 'f(b_k)', key: 'info.fb' },
  ],
  regula_falsi: [{ tex: '\\text{step}', key: 'info.step' }, STEP_Q, WIDTH_Q],
  illinois: [{ tex: 'm', key: 'info.scale' }, STEP_Q, WIDTH_Q],
  pegasus: [{ tex: 'm', key: 'info.scale' }, STEP_Q, WIDTH_Q],
  anderson_bjorck: [{ tex: 'm', key: 'info.scale' }, STEP_Q, WIDTH_Q],
  ridders: [{ tex: 'Q = e^{\\lambda (m-a)}', key: 'info.exp_factor' }, STEP_Q, WIDTH_Q],
  brent: [{ tex: '\\text{step}', key: 'info.kind' }, { tex: '|d_k|', key: 'stepSize' }, WIDTH_Q],
  chandrupatla: [
    { tex: 't', key: 'info.t' },
    { tex: '\\xi', key: 'info.xi' },
    { tex: '\\Phi', key: 'info.phi' },
  ],
  itp: [{ tex: '\\delta', key: 'info.delta' }, { tex: 'r_k', key: 'info.r' }, WIDTH_Q],
  newton: [{ tex: "f'(x_{k-1})", key: 'info.fprime' }, STEP_Q, { tex: '\\rho_k', key: 'info.rho' }],
  secant: [
    { tex: '\\text{slope}', key: 'info.chord_slope' },
    STEP_Q,
    { tex: '\\rho_k', key: 'info.rho' },
  ],
  halley: [
    { tex: "f''(x_{k-1})", key: 'info.fsecond' },
    STEP_Q,
    { tex: '\\rho_k', key: 'info.rho' },
  ],
  steffensen: [{ tex: "\\hat f'", key: 'info.slope' }, STEP_Q, { tex: '\\rho_k', key: 'info.rho' }],
  muller: [{ tex: 'b^2 - 4ac', key: 'info.disc' }, STEP_Q, { tex: '\\rho_k', key: 'info.rho' }],
  inverse_quadratic_interpolation: [
    { tex: 'a', key: 'info.ip_a' },
    STEP_Q,
    { tex: '\\rho_k', key: 'info.rho' },
  ],
  fixed_point: [{ tex: '\\lambda', key: 'info.lam' }, STEP_Q, { tex: '\\rho_k', key: 'info.rho' }],
};

/**
 * Step k with the derived quantities the card shows: the new bracket width, f′ and f″ used by
 * the step, the chord's slope, Steffensen's probe length, Müller's discriminant and the observed
 * rate ρ_k = |x_k − x_{k−1}| / |x_{k−1} − x_{k−2}| (→ |g′(x⋆)| for linear convergence).
 */
export function cardStep(method: string, trace: readonly Step[], k: number): Step | undefined {
  const s = trace[k];
  if (!s) return undefined;
  const info = { ...s.info };
  const nb = info.new_bracket as number[] | undefined;
  if (nb) info.width = nb[1] - nb[0];
  const tg = info.tangent as { slope: number } | undefined;
  if (tg) info.fprime = tg.slope;
  const der = info.derivatives as number[] | undefined;
  if (der) info.fsecond = der[2];
  const ch = info.chord as number[][] | null | undefined;
  if (ch && method === 'secant') info.chord_slope = (ch[1][1] - ch[0][1]) / (ch[1][0] - ch[0][0]);
  if (ch && method === 'steffensen') info.probe = ch[0][1];
  const par = info.parabola as { a: number; b: number; c: number } | undefined;
  if (par) info.disc = par.b * par.b - 4 * par.a * par.c;
  const ip = info.inverse_parabola as { a: number } | undefined;
  if (ip) info.ip_a = ip.a;
  if (method === 'brent') info.kind = info.step === 'inverse_quadratic' ? 'IQI' : info.step;
  if (k >= 2 && s.stepSize && trace[k - 1].stepSize)
    info.rho = s.stepSize / (trace[k - 1].stepSize as number);
  return { ...s, info };
}
