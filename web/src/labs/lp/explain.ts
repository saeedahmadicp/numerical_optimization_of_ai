/**
 * The words and formulas of a step: variable labels in KaTeX, the update rule of the current
 * step with its numbers filled in (shown in the MethodCard), and which boundary of the polygon
 * a tableau variable stands for. Pure; tested in geometry.test.ts.
 */
import type { LinearProgram, Step } from '../../core/types';
import { sig } from '../../core/format';
import { objectiveRows, type TableauInfo } from './geometry';

// ── Numbers and labels ─────────────────────────────────────────────────────────────────────

/** A number for KaTeX: 4 significant digits, a × 10^{e} outside [10⁻³, 10⁵), no −0. */
export function texNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (Math.abs(v) < 1e-12) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(digits - 1).split('e');
    const mant = m.replace(/\.?0+$/, '');
    return `${mant === '1' ? '' : mant === '-1' ? '-' : `${mant} \\times `}10^{${Number(e)}}`;
  }
  const s = v.toPrecision(digits);
  return s.includes('.') && !s.includes('e') ? s.replace(/\.?0+$/, '') : String(Number(s));
}

/**
 * A number for canvas labels and tables: integers exactly (10000, not 1.000e+4), otherwise
 * `digits` significant figures, U+2212 minus, scientific outside [10⁻³, 10⁵).
 */
export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return sig(v, digits);
  const a = Math.abs(v);
  if (a < 1e7 && Math.abs(v - Math.round(v)) <= 1e-9 * (1 + a))
    return String(Math.round(v) || 0).replace('-', '−');
  if (a >= 1e3 && a < 1e5) return String(Number(v.toPrecision(digits))).replace('-', '−');
  return sig(v, digits);
}

/**
 * A residual for KaTeX in fixed-width scientific notation, `9.84 \times 10^{-9}`, so a
 * column or a rule of residuals keeps its width while the values change during playback.
 */
export function texSci(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const [m, e] = v.toExponential(digits - 1).split('e');
  return `${m} \\times 10^{${Number(e)}}`;
}

export const texVec = (x: readonly number[], digits = 4) =>
  `(${x.map((v) => texNum(v, digits)).join(',\\, ')})`;

/** `x1` → `x_{1}`, `zM` → `z_M`, `rhs` → `\bar{\mathbf b}`. */
export function varTex(label: string): string {
  if (label === 'rhs') return '\\bar{b}';
  if (label === 'zM') return 'z_{M}';
  const m = /^([a-z])(\d+)$/.exec(label);
  return m ? `${m[1]}_{${m[2]}}` : label;
}

const SUB = '₀₁₂₃₄₅₆₇₈₉';
/** `x1` → `x₁` (canvas labels, aria text). */
export function varText(label: string): string {
  const m = /^([a-z])(\d+)$/.exec(label);
  return m ? m[1] + [...m[2]].map((d) => SUB[Number(d)]).join('') : label;
}

// ── Which boundary a variable is ───────────────────────────────────────────────────────────

export type Boundary =
  | { kind: 'axis'; j: number }
  | { kind: 'row'; a: number[]; b: number; index: number; eq: boolean }
  | { kind: 'cut'; index: number }
  | null;

/**
 * The boundary on which a nonbasic variable is zero: xⱼ = 0 is an axis; the slack sᵢ (or the
 * artificial aᵢ) of row i is that row's line; gᵩ is Gomory cut q. `inequalityForm` is the row
 * layout of the dual simplex (≤ rows, then A_eq, then −A_eq).
 */
export function boundaryOf(label: string, lp: LinearProgram, inequalityForm = false): Boundary {
  const m = /^([xsag])(\d+)$/.exec(label);
  if (!m) return null;
  const i = Number(m[2]) - 1;
  const mUb = lp.aUb?.length ?? 0,
    mEq = lp.aEq?.length ?? 0;
  if (m[1] === 'x') return { kind: 'axis', j: i };
  if (m[1] === 'g') return { kind: 'cut', index: i };
  if (i < mUb) return { kind: 'row', a: lp.aUb![i], b: lp.bUb![i], index: i, eq: false };
  const e = inequalityForm && m[1] === 's' ? (i - mUb) % mEq : i - mUb;
  if (e >= 0 && e < mEq)
    return { kind: 'row', a: lp.aEq![e], b: lp.bEq![e], index: mUb + e, eq: true };
  return null;
}

// ── Live rules ─────────────────────────────────────────────────────────────────────────────

const enters = (l: string) => `${varTex(l)}\\ \\text{enters}`;
const leaves = (l: string) => `${varTex(l)}\\ \\text{leaves}`;

function ratioList(info: TableauInfo, q: number, limit = 4): string {
  const nObj = objectiveRows(info);
  const N = info.col_labels.length - 1;
  const parts: string[] = [];
  (info.ratio_test ?? []).forEach((r, i) => {
    if (r === null) return;
    const row = info.tableau[nObj + i];
    parts.push(`\\tfrac{${texNum(Math.max(row[N], 0), 3)}}{${texNum(row[q], 3)}}`);
  });
  if (parts.length > limit) return `${parts.slice(0, limit).join(',\\, ')},\\, \\dots`;
  return parts.join(',\\, ');
}

function primalRule(step: Step, methodId: string, status: string | null): string {
  const info = step.info as unknown as TableauInfo & Record<string, unknown>;
  const lines: string[] = [];
  const phase = info.phase as number | undefined;
  if (
    methodId === 'revised_simplex' &&
    Array.isArray(info.duals) &&
    (info.duals as number[]).length <= 6
  )
    lines.push(
      `B^{\\top}\\mathbf{y} &= \\mathbf{c}_B: & \\mathbf{y} &= ${texVec(info.duals as number[], 3)}`,
    );
  if (phase === 1)
    lines.push(
      `\\textstyle\\sum_i a_i/\\rho_i &= ${texNum(info.infeasibility as number)} & &\\text{(phase 1)}`,
    );
  const q = info.entering;
  if (q !== null && q !== undefined) {
    const label = info.col_labels[q];
    if (info.drive_out) {
      lines.push(
        `${varTex(info.col_labels[info.leaving as number])} &= 0 & &\\Rightarrow\\ ${leaves(info.col_labels[info.leaving as number])},\\ ${enters(label)}`,
      );
    } else {
      const nObj = objectiveRows(info);
      const rc =
        nObj === 2
          ? `${texNum(info.tableau[0][q])}\\,M ${info.tableau[1][q] < 0 ? '-' : '+'} ${texNum(Math.abs(info.tableau[1][q]))}`
          : texNum(info.tableau[0][q]);
      lines.push(`\\bar c_{${varTex(label)}} &= ${rc} < 0 & &\\Rightarrow\\ ${enters(label)}`);
      if (info.leaving === null || info.leaving === undefined)
        lines.push(`\\bar{\\mathbf u} &\\le \\mathbf 0 & &\\Rightarrow\\ \\text{unbounded ray}`);
      else
        lines.push(
          `\\theta &= \\min\\{${ratioList(info, q)}\\} = ${texNum(nextTheta(info))} & &\\Rightarrow\\ ${leaves(info.col_labels[info.leaving])}`,
        );
    }
  } else if (info.cycling) {
    lines.push(`\\text{basis repeated} & & &\\Rightarrow\\ \\text{cycling}`);
  } else if (phase === 1 || status === 'infeasible') {
    lines.push(
      methodId === 'big_m'
        ? `\\bar{\\mathbf c} \\ge \\mathbf 0,\\ M\\text{-part} = ${texNum(info.infeasibility as number)} &> 0 & &\\Rightarrow\\ \\text{infeasible}`
        : `\\min \\textstyle\\sum a_i/\\rho_i = ${texNum(info.infeasibility as number)} &> 0 & &\\Rightarrow\\ \\text{infeasible}`,
    );
  } else {
    lines.push(
      `\\bar{\\mathbf c} &\\ge \\mathbf 0 & &\\Rightarrow\\ \\mathbf c^{\\top}\\mathbf x = ${texNum(step.fun)}\\ \\text{optimal}`,
    );
  }
  return `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`;
}

/** θ of the pivot chosen on this tableau: the minimum listed ratio. */
function nextTheta(info: TableauInfo): number {
  const r = (info.ratio_test ?? []).filter((v): v is number => v !== null);
  return r.length ? Math.min(...r) : NaN;
}

function dualRule(step: Step): string {
  const info = step.info as unknown as TableauInfo & Record<string, unknown>;
  const lines: string[] = [];
  if (info.leaving !== null && info.leaving !== undefined && info.pivot_row !== null) {
    const N = info.col_labels.length - 1;
    const r = info.pivot_row as number;
    const row = info.tableau[r];
    lines.push(
      `x_{B,r} = ${varTex(info.col_labels[info.leaving])} &= ${texNum(row[N])} < 0 & &\\Rightarrow\\ ${leaves(info.col_labels[info.leaving])}`,
    );
    if (info.entering === null)
      lines.push(`\\bar a_{rj} &\\ge 0\\ \\forall j & &\\Rightarrow\\ \\text{infeasible}`);
    else {
      const parts: string[] = [];
      (info.ratio_test ?? []).forEach((v, j) => {
        if (v !== null)
          parts.push(
            `\\tfrac{${texNum(Math.max(info.tableau[0][j], 0), 3)}}{${texNum(-row[j], 3)}}`,
          );
      });
      const best = Math.min(...(info.ratio_test ?? []).filter((v): v is number => v !== null));
      lines.push(
        `\\min_j \\tfrac{\\bar c_j}{|\\bar a_{rj}|} &= \\min\\{${parts.slice(0, 4).join(',\\,')}\\} = ${texNum(best)} & &\\Rightarrow\\ ${enters(info.col_labels[info.entering])}`,
      );
    }
  } else if (info.cycling) lines.push(`\\text{basis repeated} & & &\\Rightarrow\\ \\text{cycling}`);
  else
    lines.push(
      `\\mathbf x_B &\\ge \\mathbf 0,\\ \\bar{\\mathbf c} \\ge \\mathbf 0 & &\\Rightarrow\\ \\mathbf c^{\\top}\\mathbf x = ${texNum(step.fun)}\\ \\text{optimal}`,
    );
  return `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`;
}

function ipmRule(step: Step): string {
  const i = step.info as Record<string, number | null>;
  const head = `\\mu_{${step.k}} = \\tfrac{\\mathbf z^{\\top}\\mathbf s}{N} = ${texNum(i.mu)}`;
  if (step.k === 0 || i.sigma === null || i.sigma === undefined)
    return `\\begin{aligned} ${head} \\\\ \\text{start: Mehrotra's heuristic point} \\end{aligned}`;
  return (
    `\\begin{aligned} ${head},\\quad \\|\\mathbf r_b\\| = ${texNum(i.primal_residual)} \\\\ ` +
    `\\sigma = \\left(\\tfrac{\\mu_{\\text{aff}}}{\\mu}\\right)^{3} = ${texNum(i.sigma)},\\quad ` +
    `\\alpha^{\\text{pri}} = ${texNum(i.alpha_primal, 3)},\\ \\alpha^{\\text{dual}} = ${texNum(i.alpha_dual, 3)} \\end{aligned}`
  );
}

function affineRule(step: Step): string {
  const i = step.info as Record<string, number | null>;
  const t = `t_{${step.k}} = ${texNum(i.artificial)}`;
  if (step.k === 0 || i.alpha === null)
    return `\\begin{aligned} \\mathbf z_0 = \\mathbf e,\\ ${t} \\\\ \\text{interior start with one artificial } t \\end{aligned}`;
  return (
    `\\begin{aligned} \\mathbf z_{${step.k}} &= \\mathbf z_{${step.k - 1}} - \\alpha\\, Z^2 \\mathbf r,\\quad \\alpha = ${texNum(i.alpha)} \\\\ ` +
    `\\mathbf z^{\\top}\\mathbf r &= ${texNum(i.gap)},\\quad ${t} \\end{aligned}`
  );
}

function pdhgRule(step: Step): string {
  const i = step.info as Record<string, unknown>;
  const num = (v: unknown, d = 3) => texNum(typeof v === 'number' ? v : null, d);
  const sci = (v: unknown) => texSci(typeof v === 'number' ? v : null);
  const lines = [
    `\\tau &= \\tfrac{\\eta}{\\omega} = ${num(i.tau)},\\quad \\sigma = \\eta\\,\\omega = ${num(i.sigma)}`,
    `\\text{KKT}(\\mathbf z_{${step.k}}) &= ${sci(i.kkt_last)},\\quad \\text{KKT}(\\bar{\\mathbf z}) = ${sci(i.kkt_avg)}`,
    i.restarted
      ? `\\mathbf z_{${step.k}} &\\leftarrow \\bar{\\mathbf z}\\quad \\text{(restart ${String(i.epoch)})}`
      : `\\text{epoch } n &= ${String(i.epoch)},\\quad t = ${String(i.epoch_len)}`,
  ];
  return `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`;
}

interface BBTreeNode {
  id: number;
  branch: [number, string, number] | null;
  lp_value: number | null;
  lp_x: number[] | null;
  status: string;
}

function bbRule(step: Step, lp: LinearProgram): string {
  const info = step.info as Record<string, unknown>;
  const tree = info.tree as BBTreeNode[];
  const node = tree[info.node as number];
  const better = lp.sense === 'max' ? '\\le' : '\\ge';
  const inc = info.incumbent_value as number | null;
  const where = node.branch
    ? `x_{${node.branch[0] + 1}} ${node.branch[1] === '<=' ? '\\le' : '\\ge'} ${texNum(node.branch[2])}`
    : '\\text{root}';
  const head = `\\text{node } ${node.id}\\ (${where})`;
  let body: string;
  if (node.status === 'branched') {
    const j = info.branch_var as number;
    const v = (node.lp_x as number[])[j];
    body = `z^{\\text{LP}} = ${texNum(node.lp_value)},\\ x_{${j + 1}} = ${texNum(v)} \\notin \\mathbb Z \\Rightarrow x_{${j + 1}} \\le ${Math.floor(v)} \\,\\vee\\, x_{${j + 1}} \\ge ${Math.ceil(v)}`;
  } else if (node.status === 'integer')
    body = `z^{\\text{LP}} = ${texNum(node.lp_value)},\\ \\mathbf x \\in \\mathbb Z^{${lp.c.length}} \\Rightarrow \\text{new incumbent}`;
  else if (node.status === 'pruned_bound')
    body =
      node.lp_value === null
        ? `\\text{parent bound } ${better} z^{\\text{inc}} = ${texNum(inc)} \\Rightarrow \\text{pruned}`
        : `z^{\\text{LP}} = ${texNum(node.lp_value)} ${better} z^{\\text{inc}} = ${texNum(inc)} \\Rightarrow \\text{pruned}`;
  else if (node.status === 'infeasible') body = `\\text{LP infeasible} \\Rightarrow \\text{pruned}`;
  else body = `\\text{LP unbounded}`;
  return `\\begin{aligned} ${head} \\\\ ${body} \\end{aligned}`;
}

interface CutInfo {
  coef: number[];
  rhs: number;
  source_var: string;
  f0: number;
  f: number[];
}

/** `3x_1 + 2x_2 \le 15` for a cut coefᵀx ≤ rhs. */
export function cutTex(cut: Pick<CutInfo, 'coef' | 'rhs'>): string {
  let s = '';
  cut.coef.forEach((v, j) => {
    if (Math.abs(v) < 1e-12) return;
    const mag = Math.abs(v);
    const coef = Math.abs(mag - 1) < 1e-12 ? '' : texNum(mag);
    s += `${v < 0 ? (s ? ' - ' : '-') : s ? ' + ' : ''}${coef}x_{${j + 1}}`;
  });
  return `${s || '0'} \\le ${texNum(cut.rhs)}`;
}

function gomoryRule(step: Step, prev: Step | undefined): string {
  const info = step.info as Record<string, unknown>;
  const cut = info.cut as CutInfo | null;
  if (!cut || !prev)
    return `\\begin{aligned} \\text{LP relaxation: } \\mathbf x = ${texVec(step.x as number[])} \\\\ \\mathbf c^{\\top}\\mathbf x = ${texNum(step.fun)} \\end{aligned}`;
  const labels = prev.info.col_labels as string[];
  const terms = cut.f
    .map((f, j) => (Math.abs(f) < 1e-12 ? '' : `${texNum(f, 3)}\\,${varTex(labels[j])}`))
    .filter(Boolean)
    .slice(0, 4)
    .join(' + ');
  return (
    `\\begin{aligned} &\\textstyle\\sum_j f(\\bar a_{rj})\\, z_j \\ge f(\\bar b_r),\\quad r: ${varTex(cut.source_var)} \\\\ ` +
    `&${terms} \\ge ${texNum(cut.f0, 3)} \\iff ${cutTex(cut)} \\end{aligned}`
  );
}

/** The update rule of `step` with its numbers filled in (null: keep the static rule). */
export function liveRule(
  methodId: string,
  step: Step | undefined,
  prev: Step | undefined,
  lp: LinearProgram,
  /** `Result.extra.status` when `step` is the run's last step (else null). */
  status: string | null = null,
): string | null {
  if (!step) return null;
  try {
    switch (methodId) {
      case 'simplex':
      case 'two_phase_simplex':
      case 'big_m':
      case 'revised_simplex':
        return primalRule(step, methodId, status);
      case 'dual_simplex':
        return dualRule(step);
      case 'primal_dual_ipm':
        return ipmRule(step);
      case 'affine_scaling':
        return affineRule(step);
      case 'branch_and_bound':
        return bbRule(step, lp);
      case 'restarted_pdhg':
        return pdhgRule(step);
      case 'gomory_cuts':
        return gomoryRule(step, prev);
      default:
        return null;
    }
  } catch {
    return null;
  }
}

/** Methods whose Step.info carries a tableau. */
export const TABLEAU_METHODS = new Set([
  'simplex',
  'two_phase_simplex',
  'big_m',
  'dual_simplex',
  'revised_simplex',
  'gomory_cuts',
]);
export const INTERIOR_METHODS = new Set(['primal_dual_ipm', 'affine_scaling']);
