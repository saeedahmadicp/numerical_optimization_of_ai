/**
 * Typeset text of the regression lab: the focused method's rule with the numbers of the step
 * under the playhead, its live quantities (in β notation, not the optimization lab's 𝐱ₖ), the
 * iteration-table columns and the figure legend.
 *
 * Indexing follows Python: Step k holds β_k and, for IRLS, the weights wᵢ at β_k that the next
 * weighted solve uses to produce β_{k+1} (Step k + 1). Points are numbered 0…m − 1 in input
 * order, as in Python's info payloads and the dataset descriptions.
 */
import { sig, sci } from '../../core/format';
import type { Step } from '../../core/types';
import type { Kind } from './model';

/** A number in TeX: 4 significant digits; ×10ⁿ below 10⁻³ and from 10⁵. */
export function texNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [mant, exp] = v.toExponential(Math.max(0, digits - 1)).split('e');
    const ms = mant.includes('.') ? mant.replace(/0+$/, '').replace(/\.$/, '') : mant;
    return `${ms === '1' ? '' : ms === '-1' ? '-' : `${ms}\\times`}10^{${Number(exp)}}`;
  }
  return Number(v.toPrecision(digits)).toString();
}

const texVec = (v: readonly number[], digits = 4) =>
  `(${v.map((t) => texNum(t, digits)).join(',\\ ')})`;

/** ŷ(x) = β₀ + β₁x + … in TeX (long polynomials elided in the middle). */
export function texPoly(beta: readonly number[], digits = beta.length > 5 ? 3 : 4): string {
  const term = (j: number) => (j === 0 ? '' : j === 1 ? '\\,x' : `\\,x^{${j}}`);
  const parts: string[] = [];
  const idx = beta.length <= 5 ? beta.map((_, j) => j) : [0, 1, 2, -1, beta.length - 1];
  idx.forEach((j, n) => {
    if (j < 0) {
      parts.push('+ \\cdots');
      return;
    }
    const c = beta[j];
    const s = texNum(Math.abs(c), digits);
    if (n === 0) parts.push(`${c < 0 ? '-' : ''}${s}${term(j)}`);
    else parts.push(`${c < 0 ? '-' : '+'} ${s}${term(j)}`);
  });
  return `\\hat y(x) = ${parts.join(' ')}`;
}

const num = (v: unknown) => (typeof v === 'number' ? v : null);

/** The focused method's rule with this step's numbers, as display TeX (null: nothing to show). */
export function stepTex(
  kind: Kind,
  trace: readonly Step[],
  k: number,
  extras: { lam?: number; df?: number | null; eps?: number } = {},
): string | null {
  const s = trace[k];
  if (!s || !Array.isArray(s.x)) return null;
  const beta = s.x as number[];
  if (!beta.every(Number.isFinite)) return null;
  const next = trace[k + 1];
  const nb = next && Array.isArray(next.x) ? (next.x as number[]) : null;
  switch (kind) {
    case 'ols':
    case 'poly': {
      if (beta.length <= 3 && kind === 'ols') {
        const how =
          s.info.solver === 'normal_equations'
            ? '(A^{\\mathsf T}\\!A)^{-1}A^{\\mathsf T}\\mathbf{y}'
            : s.info.solver === 'svd'
              ? 'V\\Sigma^{+}U^{\\mathsf T}\\mathbf{y}'
              : 'R^{-1}Q^{\\mathsf T}\\mathbf{y}';
        return `\\begin{gathered} \\hat{\\boldsymbol\\beta} = ${how} = ${texVec(beta)} \\\\ ${texPoly(beta)} \\end{gathered}`;
      }
      return texPoly(beta);
    }
    case 'ridge':
      return texPoly(beta);
    case 'huber': {
      const scale = num(s.info.scale) ?? 0;
      const delta = num(s.info.delta) ?? 0;
      const w = (s.info.weights as number[] | undefined) ?? [];
      const down = w.filter((v) => v < 1).length;
      if (!nb)
        return `\\begin{aligned} \\boldsymbol\\beta_{${k}} &= ${texVec(beta)} \\\\ w_i^{(${k})} &= \\min\\!\\big(1,\\ ${texNum(delta * scale, 3)}/|r_i|\\big):\\ ${down} \\text{ of } ${w.length} < 1 \\end{aligned}`;
      return `\\begin{aligned} w_i^{(${k})} &= \\min\\!\\Big(1, \\frac{\\delta\\hat\\sigma}{|r_i(\\boldsymbol\\beta_{${k}})|}\\Big),\\ \\ \\delta\\hat\\sigma = ${texNum(delta, 4)}\\cdot${texNum(scale, 4)} \\\\ \\boldsymbol\\beta_{${k + 1}} &= \\arg\\min \\textstyle\\sum w_i^{(${k})} r_i^2 = ${texVec(nb)} \\end{aligned}`;
    }
    case 'lad': {
      const eps = extras.eps ?? 1e-6;
      if (!nb)
        return `\\begin{aligned} \\boldsymbol\\beta_{${k}} &= ${texVec(beta)} \\\\ \\textstyle\\sum_i |r_i| &= ${texNum(s.fun, 6)} \\end{aligned}`;
      return `\\begin{aligned} w_i^{(${k})} &= 1/\\max\\big(|r_i(\\boldsymbol\\beta_{${k}})|,\\ ${texNum(eps, 2)}\\big) \\\\ \\boldsymbol\\beta_{${k + 1}} &= \\arg\\min \\textstyle\\sum w_i^{(${k})} r_i^2 = ${texVec(nb)} \\end{aligned}`;
    }
    case 'theil': {
      const slopes = (s.info.slopes as number[] | undefined) ?? [];
      return `\\begin{aligned} \\beta_1 &= \\operatorname{med}\\{${slopes.length} \\text{ pairwise slopes}\\} = ${texNum(beta[1], 4)} \\\\ \\beta_0 &= \\operatorname{med}_i\\,(y_i - ${texNum(beta[1], 4)}\\,x_i) = ${texNum(beta[0], 4)} \\end{aligned}`;
    }
    case 'minimax': {
      const ref = ((s.info.reference as number[] | undefined) ?? []).map((i) => `x_{${i}}`);
      const h = num(s.info.level);
      const dev = num(s.info.max_deviation);
      const ent = s.info.entering;
      const tail =
        typeof ent === 'number'
          ? `${texNum(dev, 4)} > |h_{${k}}|:\\ x_{${ent}} \\text{ enters}`
          : `${texNum(dev, 4)} = |h_{${k}}|:\\ \\text{leveled}`;
      return `\\begin{aligned} \\{${ref.join(', ')}\\}&:\\ \\beta_0 + \\beta_1 x_{r_j} + (-1)^j h = y_{r_j} \\\\ h_{${k}} &= ${texNum(h, 4)} \\\\ \\textstyle\\max_i |r_i| &= ${tail} \\end{aligned}`;
    }
  }
}

/** Live quantities in β notation (label TeX, value text). */
export function stepQuantities(
  kind: Kind,
  step: Step | undefined,
  extras: { lam?: number; df?: number | null; floorCount?: number } = {},
): { tex: string; value: string; wide?: boolean }[] {
  if (!step) return [];
  const beta = Array.isArray(step.x) ? (step.x as number[]) : [];
  const show = (v: number | null | undefined, d = 5) =>
    v === null || v === undefined || Number.isNaN(v)
      ? '—'
      : Math.abs(v) < 1e-3 && v !== 0
        ? sci(v, 3)
        : Math.abs(v) >= 1e5
          ? sci(v, 3)
          : sig(v, d);
  const out: { tex: string; value: string; wide?: boolean }[] = [
    { tex: 'k', value: String(step.k) },
  ];
  out.push({
    tex: '\\boldsymbol\\beta_k',
    value:
      beta.length <= 3
        ? `(${beta.map((b) => show(b, 4)).join(', ')})`
        : `${beta.length} coefficients`,
    wide: beta.length > 1,
  });
  const fTex: Record<Kind, string> = {
    ols: '\\textstyle\\sum r_i^2',
    poly: '\\textstyle\\sum r_i^2',
    ridge: '\\textstyle\\sum r_i^2 + \\lambda\\|\\boldsymbol\\beta_{1:}\\|^2',
    huber: '\\textstyle\\sum\\rho_\\delta(r_i/\\hat\\sigma)',
    lad: '\\textstyle\\sum|r_i|',
    theil: '\\textstyle\\sum r_i^2',
    minimax: '\\max_i|r_i|',
  };
  out.push({ tex: fTex[kind], value: show(step.fun) });
  const info = step.info;
  switch (kind) {
    case 'ols':
    case 'poly':
      out.push({ tex: '\\kappa_2(A)', value: show(num(info.cond), 4) });
      out.push({ tex: '\\operatorname{rank} A', value: String(info.rank ?? '—') });
      break;
    case 'ridge':
      out.push({ tex: '\\lambda', value: show(extras.lam ?? num(info.lambda), 3) });
      out.push({ tex: '\\operatorname{tr} H_\\lambda', value: show(extras.df ?? null, 4) });
      break;
    case 'huber': {
      const w = (info.weights as number[] | undefined) ?? [];
      out.push({ tex: '\\hat\\sigma', value: show(num(info.scale), 4) });
      out.push({
        tex: '\\#\\{w_i < 1\\}',
        value: `${w.filter((v) => v < 1).length} of ${w.length}`,
      });
      out.push({ tex: '\\|\\Delta\\boldsymbol\\theta\\|_\\infty', value: show(step.stepSize, 3) });
      break;
    }
    case 'lad':
      out.push({
        tex: 'S_\\varepsilon(\\boldsymbol\\beta_k)',
        value: show(num(info.smoothed_objective)),
      });
      out.push({ tex: '\\#\\{|r_i| \\le \\varepsilon\\}', value: String(extras.floorCount ?? 0) });
      out.push({ tex: '\\|\\Delta\\boldsymbol\\theta\\|_\\infty', value: show(step.stepSize, 3) });
      break;
    case 'theil':
      out.push({
        tex: '\\#\\text{pairs}',
        value: String(((info.slopes as number[]) ?? []).length),
      });
      break;
    case 'minimax':
      out.push({ tex: 'h_k', value: show(num(info.level), 5) });
      out.push({
        tex: '\\text{reference}',
        value: ((info.reference as number[]) ?? []).join(', '),
      });
      break;
  }
  return out;
}
