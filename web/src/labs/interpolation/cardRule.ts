/**
 * The MethodCard's update rule with this step's numbers filled in: the general rule on the
 * first line, the instance that this step computed on the second (TeX `aligned`).
 */
import type { Result } from '../../core/types';
import { DOCS } from '../../methods/interpolation/methods';
import { RATIONAL_DOCS } from '../../methods/interpolation/rational';

/** A number for TeX: 4 significant digits, `1.2\times10^{-5}` outside [10⁻³, 10⁵). */
export function texNum(v: unknown, digits = 4): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(digits - 1).split('e');
    return `${m.replace(/\.?0+$/, '')}\\times10^{${Number(e)}}`;
  }
  const s = v.toPrecision(digits);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** `(x - x_j)` with the node's value: `(x + 0.8)`; the node 0 gives the bare factor `x`. */
function factor(xj: number): string {
  if (xj === 0) return 'x';
  return xj < 0 ? `(x + ${texNum(-xj)})` : `(x - ${texNum(xj)})`;
}

/**
 * ∏ (x − x_j) with the node values filled in. A bare `x` factor is set off by thin spaces, so
 * it can never fuse with a neighboring control word (`\cdots x`, not `\cdotsx`).
 */
export function product(nodes: readonly number[], max = 3): string {
  if (nodes.length === 0) return '1';
  const shown =
    nodes.length <= max
      ? nodes.map(factor)
      : [...nodes.slice(0, 2).map(factor), '\\cdots', factor(nodes[nodes.length - 1])];
  return shown.reduce((acc, f, i) => {
    if (i === 0) return f;
    const prev = shown[i - 1];
    const gap =
      f === 'x' || prev === 'x' || f.startsWith('\\') || prev.startsWith('\\') ? '\\,' : '';
    return `${acc}${gap}${f}`;
  }, '');
}

/** ` + v` or ` - |v|` for a term that follows another one. */
function signed(v: number): string {
  return `${v < 0 ? '-' : '+'} ${texNum(Math.abs(v))}`;
}

const nums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);

/** Split a rule at its top-level `\\quad` / `\\qquad` separators (one clause per line). */
export function clauses(tex: string): string[] {
  const out: string[] = [];
  let depth = 0,
    from = 0;
  for (let i = 0; i < tex.length; i++) {
    const c = tex[i];
    if (c === '{') depth++;
    else if (c === '}') depth--;
    else if (depth === 0 && c === '\\' && /^\\q?quad(?![a-zA-Z])/.test(tex.slice(i))) {
      const len = tex.startsWith('\\qquad', i) ? 6 : 5;
      out.push(tex.slice(from, i).replace(/,\s*$/, '').trim());
      from = i + len;
      i += len - 1;
    }
  }
  out.push(tex.slice(from).trim());
  return out.filter(Boolean);
}

/**
 * The general rule, then the instance this step computed, one clause per line (TeX `aligned`):
 * the MethodCard column is narrow, and a rule that fits needs no scrolling.
 */
function aligned(general: string, instance: string): string {
  const rows = [...clauses(general), ...clauses(instance)];
  return `\\begin{aligned}${rows.map((r, i) => `${i ? '\\\\[2pt]' : ''}&${r}`).join('')}\\end{aligned}`;
}

/**
 * The rule for `methodId` at step k of `result`; `nodes`/`values` are the data in the method's
 * order. Falls back to the static rule when the step has nothing to fill in.
 */
export function filledRule(
  methodId: string,
  result: Result,
  k: number,
  nodes: readonly number[],
  values: readonly number[],
): string {
  const rule =
    (DOCS[methodId] ?? RATIONAL_DOCS[methodId as keyof typeof RATIONAL_DOCS])?.rule ?? '';
  const s = result.trace[k];
  if (!s) return rule;
  const info = s.info;
  const xk = nodes[k],
    yk = values[k];
  switch (methodId) {
    case 'lagrange':
      return aligned(
        'p_k(x) = p_{k-1}(x) + y_k\\,\\ell_k(x),\\quad \\ell_k(x) = \\prod_{m \\ne k} \\frac{x - x_m}{x_k - x_m}',
        `p_{${k}}(x) = ${k === 0 ? texNum(yk) : `p_{${k - 1}}(x) ${signed(yk)}`}\\,\\ell_{${k}}(x),\\quad \\ell_{${k}}(${texNum(xk)}) = 1`,
      );
    case 'barycentric': {
      const w = nums(s.x);
      return aligned(
        'w_k = \\frac{1}{\\prod_{m<k}(x_k - x_m)},\\quad w_j \\leftarrow \\frac{w_j}{x_j - x_k}\\ (j<k)',
        k === 0
          ? `w_0 = 1,\\quad x_0 = ${texNum(xk)}`
          : `w_{${k}} = \\frac{1}{\\prod_{m<${k}}(${texNum(xk)} - x_m)} = ${texNum(w[k])}`,
      );
    }
    case 'newton_divided_differences': {
      const row = nums(info.new_row);
      const c = row[k];
      const args = k === 0 ? 'x_0' : k === 1 ? 'x_0, x_1' : `x_0,\\dots,x_{${k}}`;
      return aligned(
        'p_k(x) = p_{k-1}(x) + f[x_0,\\dots,x_k]\\prod_{j<k}(x - x_j)',
        k === 0
          ? `p_0(x) = f[x_0] = ${texNum(c)}`
          : `p_{${k}}(x) = p_{${k - 1}}(x) + \\underbrace{${c < 0 ? `(${texNum(c)})` : texNum(c)}}_{\\mathclap{\\textstyle\\footnotesize f[${args}]}}\\,${product(nodes.slice(0, k))}`,
      );
    }
    case 'neville': {
      const col = nums(info.column);
      const xs = info.x_eval as number;
      return aligned(
        'Q_{i,j} = \\frac{(x^\\ast - x_{i-j})\\,Q_{i,j-1} - (x^\\ast - x_i)\\,Q_{i-1,j-1}}{x_i - x_{i-j}}',
        `x^\\ast = ${texNum(xs)}:\\quad Q_{${k},${k}} = P_{0..${k}}(x^\\ast) = ${texNum(col[0])}` +
          (col.length > 1
            ? `,\\ \\ Q_{${nodes.length - 1},${k}} = ${texNum(col[col.length - 1])}`
            : ''),
      );
    }
    case 'chebyshev_interpolation': {
      const c = nums(s.x);
      const N = (result.extra.coefficients as number[] | undefined)?.length ?? c.length;
      return aligned(
        'p(x) = \\sum_{k=0}^{N-1} c_k T_k(u),\\quad u = \\tfrac{2x - (a+b)}{b-a}',
        `c_{${k}} = ${texNum(c[k])},\\quad p_{${k}} = \\sum_{j\\le ${k}} c_j T_j(u),\\quad N = ${N}`,
      );
    }
    case 'linear_spline': {
      const m = nums(info.secants);
      return aligned(
        'S_i(x) = y_i + m_i\\,(x - x_i)',
        `S_0(x) = ${texNum(values[0])} ${signed(m[0])}\\,${factor(nodes[0])},\\ \\dots`,
      );
    }
    case 'aaa': {
      // Step 0 is the constant mean; step m adds z_m and refits w on the other samples.
      const node = nums(info.node);
      const err = info.sample_error as number;
      if (node.length !== 2)
        return aligned(
          'r_0 = \\operatorname{mean}_Z f',
          `r_0 = ${texNum(nums(info.curve)[0])},\\quad \\max_Z|f - r_0| = ${texNum(err)}`,
        );
      const nReal = Number(info.n_interval_poles ?? 0);
      return aligned(
        'z_m = \\arg\\max_{Z}|f - r_{m-1}|,\\quad \\mathbf{w} = \\arg\\min_{\\|\\mathbf{w}\\|=1}\\|A^{(m)}\\mathbf{w}\\|',
        `z_{${k}} = ${texNum(node[0])},\\quad \\sigma_{\\min} = ${texNum(info.sigma_min)},\\quad \\max_Z|f - r_{${k}}| = ${texNum(err)}` +
          (nReal
            ? `,\\quad ${nReal}\\ \\text{real pole${nReal === 1 ? '' : 's'} in}\\ [a, b]`
            : ''),
      );
    }
    case 'floater_hormann': {
      const win = nums(info.window);
      const d = result.extra.d as number;
      const w = info.weight as number;
      return aligned(
        'w_k = (-1)^{k-d}\\sum_{i \\in J_k}\\prod_{j=i,\\,j\\ne k}^{i+d}\\frac{h}{|x_k - x_j|}',
        `d = ${d},\\quad J_{${k}} = \\{${win[0]}, \\dots, ${win[1]}\\},\\quad w_{${k}} = ${texNum(w)}`,
      );
    }
    case 'pchip':
      return pchipRule(s.info, nodes);
    default:
      return splineRule(methodId, s.info, rule);
  }
}

function splineRule(methodId: string, info: Record<string, unknown>, rule: string): string {
  const stage = info.stage as string;
  const n = nums(info.diag).length || nums(info.slopes).length;
  if (stage === 'assemble') {
    const diag = nums(info.diag),
      rhs = nums(info.rhs);
    return aligned(
      rule,
      `T\\mathbf{s} = \\mathbf{r},\\ T \\in \\mathbb{R}^{${n}\\times${n}}\\ \\text{tridiagonal},\\quad T_{1,1} = ${texNum(diag[1])},\\ r_1 = ${texNum(rhs[1])}`,
    );
  }
  if (stage === 'forward_sweep') {
    const piv = nums(info.pivots),
      mult = nums(info.multipliers);
    return aligned(
      'l_i = \\frac{T_{i,i-1}}{u_{i-1}},\\quad u_i = T_{ii} - l_i T_{i-1,i},\\quad z_i = r_i - l_i z_{i-1}',
      `l_1 = ${texNum(mult[0])},\\quad u_1 = ${texNum(piv[1])},\\quad u_{${piv.length - 1}} = ${texNum(piv[piv.length - 1])}`,
    );
  }
  if (stage === 'back_substitution') {
    const s = nums(info.slopes);
    return aligned(
      's_{n-1} = \\frac{z_{n-1}}{u_{n-1}},\\quad s_i = \\frac{z_i - T_{i,i+1}\\,s_{i+1}}{u_i}',
      `s_0 = ${texNum(s[0])},\\quad s_1 = ${texNum(s[1])},\\ \\dots,\\ s_{${s.length - 1}} = ${texNum(s[s.length - 1])}`,
    );
  }
  const coef = (info.coefficients as number[][] | undefined)?.[0];
  void methodId;
  if (!coef) return rule;
  return aligned(
    'S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3',
    `S_0:\\ a = ${texNum(coef[0])},\\ b = ${texNum(coef[1])},\\quad c = ${texNum(coef[2])},\\ d = ${texNum(coef[3])}`,
  );
}

function pchipRule(info: Record<string, unknown>, nodes: readonly number[]): string {
  const stage = info.stage as string;
  if (stage === 'secants') {
    const m = nums(info.secants);
    return aligned(
      'm_i = \\frac{y_{i+1} - y_i}{x_{i+1} - x_i}',
      `m_0 = ${texNum(m[0])},\\ m_1 = ${texNum(m[1])},\\ \\dots,\\ m_{${m.length - 1}} = ${texNum(m[m.length - 1])}`,
    );
  }
  if (stage === 'slopes') {
    const s = nums(info.slopes);
    const limited = (info.limited as boolean[] | undefined) ?? [];
    const nLim = limited.filter(Boolean).length;
    return aligned(
      '\\frac{w_1 + w_2}{s_k} = \\frac{w_1}{m_{k-1}} + \\frac{w_2}{m_k},\\quad s_k = 0 \\text{ if } m_{k-1} m_k \\le 0',
      `s_1 = ${texNum(s[1])},\\ \\dots;\\ ${nLim}\\ \\text{of}\\ ${s.length}\\ \\text{slopes limited}`,
    );
  }
  const coef = (info.coefficients as number[][] | undefined)?.[0];
  if (!coef) return DOCS.pchip.rule;
  return aligned(
    'S_i(x) = y_i + s_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3',
    `S_0:\\ x_0 = ${texNum(nodes[0])},\\ s_0 = ${texNum(coef[1])},\\quad c_0 = ${texNum(coef[2])},\\ d_0 = ${texNum(coef[3])}`,
  );
}
