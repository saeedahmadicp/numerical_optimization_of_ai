/**
 * The update rule of the step on screen with its numbers filled in, as KaTeX:
 *
 *   SGD        𝐰₅₇ = (𝐰₅₆) − η (𝐠) = (𝐰₅₇)
 *   momentum   𝐰ₖ = (𝐰ₖ₋₁) + (βvₖ₋₁) − η (𝐠) = (𝐰ₖ)
 *   adaptive   𝐰ₖ = (𝐰ₖ₋₁) − (η/(√r + ε)) ⊙ (𝐠 or m̂) = (𝐰ₖ)
 *   SVRG/SAGA/SAG  𝐰ₖ = (𝐰ₖ₋₁) − η (𝐯) = (𝐰ₖ)
 *
 * Every number comes from the recorded step (`x`, `stepSize`, `info.update`, `info.stoch_grad`,
 * `info.scaled_lr`, `info.m_hat`), so the identity holds to rounding of the 4 printed digits.
 */
import type { Step } from '../../core/types';
import { momentumPush, ruleKind } from './geometry';

/** 4 significant digits; ×10ⁿ outside [10⁻³, 10⁵); ASCII minus (KaTeX sets it as −). */
export function texNum(v: number, digits = 4): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? '\\text{NaN}' : v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(digits - 1).split('e');
    const mant = m.replace(/\.?0+$/, '');
    return `${mant}\\times10^{${Number(e)}}`;
  }
  return String(Number(v.toPrecision(digits)));
}

export const texVec = (v: readonly number[]) =>
  `\\begin{pmatrix}${v.map((x) => texNum(x)).join('\\\\')}\\end{pmatrix}`;

/**
 * A vector with its name under a brace. KaTeX sets an \underbrace label in script style, so
 * its own subscripts would fall to scriptscript (about 8.5 px); \textstyle\footnotesize keeps
 * the label at 0.8 em with its subscripts near 9.4 px.
 */
const brace = (body: string, label: string) =>
  `\\underbrace{${body}}_{\\textstyle\\footnotesize ${label}}`;

const isVec = (v: unknown): v is number[] =>
  Array.isArray(v) && v.length === 2 && v.every((x) => typeof x === 'number');

/** The filled-in update for `step` (k ≥ 1), or null at k = 0 / when info is missing. */
export function stepTex(methodId: string, step: Step): string | null {
  const { x, info, stepSize: eta } = step;
  const update = info.update,
    g = info.stoch_grad;
  if (step.k === 0 || !isVec(x) || !isVec(update) || !isVec(g) || eta === null) return null;
  const base = [x[0] - update[0], x[1] - update[1]];
  const lhs = `\\mathbf{w}_{${step.k}}`;
  const kind = ruleKind(methodId);
  const tail = ` = ${texVec(x)}`;
  if (kind === 'momentum' || kind === 'nesterov') {
    const push = momentumPush(step.k, update, g, eta);
    const at = kind === 'nesterov' ? '\\tilde{\\mathbf{w}}' : '\\mathbf{w}';
    return (
      `${lhs} = ${texVec(base)} + ${brace(texVec(push), '\\beta\\mathbf{v}')}` +
      ` - ${texNum(eta)}${brace(texVec(g), `\\nabla f_B(${at})`)}${tail}`
    );
  }
  if (kind === 'adaptive') {
    const scaled = info.scaled_lr;
    const dir = methodId === 'stochastic_adam' ? info.m_hat : g;
    if (!isVec(scaled) || !isVec(dir)) return null;
    const dirName = methodId === 'stochastic_adam' ? '\\hat{\\mathbf{m}}' : '\\mathbf{g}';
    return (
      `${lhs} = ${texVec(base)} - ${brace(texVec(scaled), '\\eta/(\\sqrt{\\cdot}+\\varepsilon)')}` +
      ` \\odot ${brace(texVec(dir), dirName)}${tail}`
    );
  }
  const name = kind === 'sgd' ? '\\mathbf{g}' : '\\mathbf{v}';
  return `${lhs} = ${texVec(base)} - ${texNum(eta)}${brace(texVec(g), name)}${tail}`;
}
