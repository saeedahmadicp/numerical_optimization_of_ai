/** Plain-text and TeX helpers for the least-squares lab (no React). */
import { sig } from '../../core/format';
import type { Step } from '../../core/types';
import type { DampingSeries } from './DampingChart';
import type { MethodKind } from './models';

/** A number as TeX: 4 significant digits, `1.23\\times10^{-8}` outside [10⁻³, 10⁵), `\\infty`. */
export function texNum(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '\\text{—}';
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : '-\\infty';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a < 1e-3 || a >= 1e5) {
    const [mant, exp] = x.toExponential(digits - 1).split('e');
    const mt = mant.replace(/\.?0+$/, '');
    const e = Number(exp);
    return mt === '1' ? `10^{${e}}` : mt === '-1' ? `-10^{${e}}` : `${mt}\\times10^{${e}}`;
  }
  const s = x.toPrecision(digits);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** A 2-vector as TeX: `(0.4123,\\,-1.2)`. */
export function texVec(x: readonly number[] | null | undefined, digits = 4): string {
  if (!x) return '\\text{—}';
  return `(${x.map((v) => texNum(v, digits)).join(',\\,')})`;
}

/** One sentence for screen readers: the μ range and the rejected trials of each method. */
export function describeDamping(series: readonly DampingSeries[]): string {
  return series
    .map((s) => {
      const mus = s.mu.filter((v): v is number => v !== null && v > 0);
      const rejected = s.accepted.filter((a) => a === false).length;
      const muText = mus.length
        ? `μ from ${sig(mus[0], 2)} to ${sig(mus[mus.length - 1], 2)}`
        : 'undamped (μ = 0)';
      return `${s.name}: ${muText}, ${rejected} rejected trial${rejected === 1 ? '' : 's'}`;
    })
    .join('; ');
}

const EPS = 2.220446049250313e-16;

/**
 * Indices of the components of 𝐱ₖ that cancelled to rounding level in the last step:
 * nonzero, and below 64 ε times the size of the previous iterate's component (or 10⁻¹²).
 */
export function roundingNoise(x: readonly number[], prev?: readonly number[]): number[] {
  const out: number[] = [];
  x.forEach((v, j) => {
    const ref = Math.max(1, Math.abs(prev?.[j] ?? 0));
    if (v !== 0 && Number.isFinite(v) && Math.abs(v) <= 64 * EPS * ref) out.push(j);
  });
  return out;
}

/** 𝒪(10ᵉ) for a value known only to its order of magnitude. */
export function orderTex(v: number): string {
  return `\\mathcal{O}(10^{${Math.floor(Math.log10(Math.abs(v)))}})`;
}

/** 𝐱 as TeX, with components that cancelled to rounding level against `prev` as 𝒪(10ᵉ). */
export function texIterate(x: readonly number[], prev?: readonly number[], digits = 4): string {
  const noise = roundingNoise(x, prev);
  return `(${x.map((v, j) => (noise.includes(j) ? orderTex(v) : texNum(v, digits))).join(',\\,')})`;
}

/**
 * The TeX of the substituted rule (step taken from 𝐱ₖ to 𝐱ₖ₊₁). `names` are the two parameter
 * names (a, b; V, K; x, y), used when the rule has to talk about one component.
 */
export function stepRuleTex(
  kind: MethodKind,
  trace: readonly Step[],
  k: number,
  names: readonly [string, string] = ['x_1', 'x_2'],
): string | null {
  const cur = trace[k];
  if (!cur) return null;
  const next = trace[k + 1];
  const i = String(k),
    j = String(k + 1);
  if (!next) {
    const x = cur.x as number[];
    const prev = trace[k - 1]?.x as number[] | undefined;
    const noise = roundingNoise(x, prev);
    if (noise.length) {
      // A component that cancelled to rounding level: its sign, and everything that depends
      // on it (f, ‖∇f‖), is not reproducible — Python and this port land on different noise.
      const which = noise.map((j) => `$${names[j]}_{${i}}$`).join(' and ');
      return (
        '\\begin{aligned}' +
        `\\mathbf{x}_{${i}} &= ${texIterate(x, prev)}\\\\` +
        `&\\text{${which} cancelled to rounding level: the sign is noise,}\\\\` +
        `&\\text{and so are } f(\\mathbf{x}_{${i}}) \\text{ and } \\nabla f(\\mathbf{x}_{${i}})` +
        '\\end{aligned}'
      );
    }
    return (
      '\\begin{aligned}' +
      `\\mathbf{x}_{${i}} &= ${texVec(x)}\\\\` +
      `f(\\mathbf{x}_{${i}}) &= ${texNum(cur.fun)}\\\\` +
      `\\|\\nabla f(\\mathbf{x}_{${i}})\\|_2 &= \\|J_{${i}}^{\\mathsf T}\\mathbf{r}_{${i}}\\|_2 = ${texNum(cur.gradNorm, 3)}` +
      '\\end{aligned}'
    );
  }
  const info = next.info;
  const step = info.step as number[] | null;
  const gain = info.gain_ratio as number | null;
  if (kind === 'gauss_newton') {
    const alpha = (info.alpha as number | null) ?? 1;
    const trials = ((info.trials as unknown[] | undefined) ?? []).length;
    const halvings = Math.round(-Math.log2(alpha));
    const alphaTex = halvings === 0 ? '1' : `2^{-${halvings}}`;
    const trialText = trials > 1 ? `\\ \\text{(${trials} trials)}` : '';
    return (
      '\\begin{aligned}' +
      `\\mathbf{p}_{${i}} &= -J_{${i}}^{+}\\mathbf{r}_{${i}} = ${texVec(step)}\\\\` +
      `\\alpha_{${i}} &= ${alphaTex}${trialText},\\quad \\varrho_{${i}} = ${texNum(gain, 3)}\\\\` +
      `\\mathbf{x}_{${j}} &= \\mathbf{x}_{${i}} + \\alpha_{${i}}\\mathbf{p}_{${i}} = ${texIterate(next.x as number[], cur.x as number[])}` +
      '\\end{aligned}'
    );
  }
  const mu = info.lambda as number;
  const accepted = info.accepted === true;
  const nuPrev = (cur.info.nu as number | undefined) ?? 2;
  const norm = step ? Math.hypot(...step) : null;
  const decision = accepted
    ? `\\varrho_{${i}} &= ${texNum(gain, 3)} > 0 \\;\\Rightarrow\\; \\text{accept: } \\mathbf{x}_{${j}} = \\mathbf{x}_{${i}} + \\mathbf{h}_{${i}}\\\\` +
      `\\mu_{${j}} &= \\mu_{${i}}\\max\\{\\tfrac13,\\,1-(2\\varrho_{${i}}-1)^3\\} = ` +
      texNum(mu * Math.max(1 / 3, 1 - (2 * (gain as number) - 1) ** 3), 3)
    : `\\varrho_{${i}} &= ${gain === null ? '\\text{undefined}' : texNum(gain, 3)} \\le 0 ` +
      `\\;\\Rightarrow\\; \\text{reject: } \\mathbf{x}_{${j}} = \\mathbf{x}_{${i}}\\\\` +
      `\\mu_{${j}} &= \\nu\\,\\mu_{${i}} = ${texNum(nuPrev, 3)}\\cdot${texNum(mu, 3)} = ${texNum(nuPrev * mu, 3)}`;
  return (
    '\\begin{aligned}' +
    `\\mu_{${i}} &= ${texNum(mu, 3)},\\quad \\|\\mathbf{h}_{${i}}\\| = ${texNum(norm, 3)}\\\\` +
    // The step's definition and its value on two rows (one row is wider than the Details column).
    `\\mathbf{h}_{${i}} &= -(J_{${i}}^{\\mathsf T}J_{${i}} + \\mu_{${i}} I)^{-1}J_{${i}}^{\\mathsf T}\\mathbf{r}_{${i}}\\\\` +
    `&= ${texVec(step, 3)}\\\\` +
    decision +
    '\\end{aligned}'
  );
}
