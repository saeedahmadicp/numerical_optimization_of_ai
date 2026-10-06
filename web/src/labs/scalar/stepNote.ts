/**
 * "This step" in KaTeX with the numbers of the current step filled in: the comparison that
 * decides the next cut, the vertex that becomes the next trial, Newton's quotient. Pure (TeX
 * strings), unit-tested in tests/scalar/geometry.test.ts.
 *
 * Every equation printed here is the one the method evaluated: when a safeguard replaces the
 * textbook point (Brent's minimum step tol, Fibonacci's ε-step, a rejected vertex), the note
 * prints the textbook point and the replacement on separate lines.
 */
import type { Result, Step } from '../../core/types';
import { RHO } from '../../methods/scalar/methods';
import { kindOf, nextView, proportions, type StepView } from './geometry';

/** A number in TeX: `digits` significant figures, ×10ⁿ outside [10⁻³, 10⁵), −∞/∞/—. */
export function texNum(v: number | null | undefined, digits = 5): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '\\text{—}';
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(Math.max(0, Math.min(digits, 6) - 1)).split('e');
    const mant = m.includes('.') ? m.replace(/0+$/, '').replace(/\.$/, '') : m;
    const pow = `10^{${Number(e)}}`;
    if (mant === '1') return pow;
    if (mant === '-1') return `-${pow}`;
    return `${mant} \\times ${pow}`;
  }
  const s = v.toPrecision(Math.min(17, digits));
  return s.includes('.') && !s.includes('e') ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
}

/** Significant digits that tell apart points `width` apart near `scale` (5 to 15). */
export function digitsFor(width: number, scale: number): number {
  if (!(width > 0) || !Number.isFinite(width)) return 5;
  const d = Math.ceil(Math.log10(Math.max(Math.abs(scale), width) / width)) + 3;
  return Math.max(5, Math.min(15, d));
}

/**
 * Significant digits that print every pair of distinct points of `xs` as different numbers:
 * digitsFor of the smallest positive gap between them (not of the bracket width, which can be
 * far larger than the spacing of the points the method keeps apart).
 */
export function digitsForPoints(xs: readonly (number | null | undefined)[], scale: number): number {
  const s = xs.filter((x): x is number => typeof x === 'number' && Number.isFinite(x));
  s.sort((p, q) => p - q);
  let gap = Infinity;
  for (let i = 1; i < s.length; i++) {
    const g = s[i] - s[i - 1];
    if (g > 0 && g < gap) gap = g;
  }
  if (!Number.isFinite(gap)) return digitsFor(Math.abs(scale) * 1e-4 || 1e-4, scale);
  return digitsFor(gap, scale);
}

/** Which ends of [a, b] the next bracket [na, nb] discards. */
export function droppedEnds(
  bracket: readonly [number, number],
  next: readonly [number, number],
): { left: boolean; right: boolean } {
  return { left: next[0] > bracket[0], right: next[1] < bracket[1] };
}

/** The last Fibonacci step (its probes are p and q = p + ε, not Fibonacci points). */
export function isEpsilonStep(methodId: string, trace: readonly Step[], k: number): boolean {
  const s = trace[k];
  return methodId === 'fibonacci_search' && !!s && typeof s.info.eps === 'number';
}

const interval = (a: number, b: number, d: number) => `[${texNum(a, d)},\\ ${texNum(b, d)}]`;
const tuple = (xs: readonly number[], d: number) => `(${xs.map((x) => texNum(x, d)).join(',\\ ')})`;

/**
 * The TeX lines for step k of a run (`aligned` body, `\\` separated), or null when there is
 * nothing to say (empty trace).
 */
export function stepNote(
  methodId: string,
  result: Result,
  views: readonly StepView[],
  k: number,
): string | null {
  const trace: readonly Step[] = result.trace;
  const v = views[k];
  if (!v) return null;
  const kind = kindOf(methodId);
  const next = nextView(methodId, trace, k);
  const nextInfo = trace[k + 1]?.info;
  const lines: string[] = [];
  const last = k === trace.length - 1;

  switch (kind) {
    case 'interval': {
      const fib = methodId === 'fibonacci_search';
      const [n1, n2] = fib ? ['\\lambda', '\\mu'] : ['x_1', 'x_2'];
      const [a, b] = v.bracket!;
      const pts = [a, b, ...v.probes.map((p) => p.x), ...(next?.nextBracket ?? [])];
      for (const [tx] of next?.trials ?? []) pts.push(tx);
      const d = digitsForPoints(pts, v.x);
      // Many digits (a narrow bracket far from 0): the two ends stacked in one row, so the block
      // stays narrow enough for the method card.
      const long = d > 8;
      const stacked = (u: number, w: number) =>
        `\\begin{array}{l}${texNum(u, d)},\\ \\\\${texNum(w, d)}\\end{array}`;
      if (long) lines.push(`a_{${k}},\\ b_{${k}} &= ${stacked(a, b)}`);
      else lines.push(`[a_{${k}},\\, b_{${k}}] &= ${interval(a, b, d)}`);
      lines.push(`b_{${k}} - a_{${k}} &= ${texNum(b - a, 4)}`);
      if (v.probes.length === 2) {
        const [p, q] = v.probes;
        const [r1, r2, r3] = proportions(a, b, p.x, q.x);
        if (long) lines.push(`${n1},\\ ${n2} &= ${stacked(p.x, q.x)}`);
        else lines.push(`${n1},\\ ${n2} &= ${texNum(p.x, d)},\\ ${texNum(q.x, d)}`);
        lines.push(`\\text{split} &= ${texNum(r1, 3)} : ${texNum(r2, 3)} : ${texNum(r3, 3)}`);
        const diff = p.f - q.f;
        if (next && next.nextBracket) {
          const [na, nb] = next.nextBracket;
          const ratio = (nb - na) / (b - a);
          if (fib && isEpsilonStep(methodId, trace, k + 1)) {
            // Last Fibonacci step: cut on λ, μ, then compare p with q = p + ε and cut again.
            const leftFirst = diff > 0;
            const pName = leftFirst ? n2 : n1;
            const pf = leftFirst ? q.f : p.f;
            const qx = next.trials[0]?.[0];
            const qf = next.trials[0]?.[1];
            const eps = nextInfo!.eps as number;
            lines.push(
              `f(${n1}) - f(${n2}) &= ${texNum(diff, 3)} ${diff > 0 ? '>' : '\\le'} 0 \\;\\Rightarrow\\; \\text{drop } ${
                leftFirst ? `[a_{${k}},\\ ${n1})` : `(${n2},\\ b_{${k}}]`
              }`,
            );
            lines.push(
              `p &= ${pName},\\quad q = p + \\varepsilon = ${texNum(qx, d)},\\ \\ \\varepsilon = ${texNum(eps, 3)}`,
            );
            const d2 = pf - (qf ?? NaN);
            const second =
              d2 > 0
                ? leftFirst
                  ? `[${n1},\\ p)`
                  : `[a_{${k}},\\ p)`
                : leftFirst
                  ? `(q,\\ b_{${k}}]`
                  : `(q,\\ ${n2}]`;
            lines.push(
              `f(p) - f(q) &= ${texNum(d2, 3)} ${d2 > 0 ? '>' : '\\le'} 0 \\;\\Rightarrow\\; \\text{drop } ${second}`,
            );
          } else {
            const keepLeft = next.stepKind === 'right';
            lines.push(
              `f(${n1}) - f(${n2}) &= ${texNum(diff, 3)} ${keepLeft ? '\\le' : '>'} 0 \\;\\Rightarrow\\; \\text{drop } ${
                keepLeft ? `(${n2},\\ b_{${k}}]` : `[a_{${k}},\\ ${n1})`
              }`,
            );
          }
          lines.push(
            `\\frac{b_{${k + 1}} - a_{${k + 1}}}{b_{${k}} - a_{${k}}} &= ${texNum(ratio, 4)}`,
          );
        } else if (last && diff === 0 && !result.converged) {
          // A stop by rounding (dichotomous search): the last comparison was a tie.
          lines.push(
            `f(${n1}) - f(${n2}) &= 0 \\quad (\\text{rounding tie: neither side is lower})`,
          );
        }
      }
      break;
    }
    case 'parabolic': {
      const u = next?.trials[0]?.[0];
      const vx = next?.vertex ?? null;
      const d = digitsForPoints([...v.probes.map((p) => p.x), u, vx], v.x);
      lines.push(
        `(x_l, x_m, x_r) &= ${tuple(
          v.probes.map((p) => p.x),
          d,
        )}`,
      );
      if (next) {
        if (next.stepKind === 'parabolic')
          lines.push(`u &= \\text{vertex} = ${texNum(u, d)} \\quad \\text{(parabolic step)}`);
        else if (next.stepKind === 'probe') {
          lines.push(
            `\\text{vertex} &= ${texNum(vx, d)} \\quad (\\text{within } \\delta \\text{ of } x_m)`,
          );
          lines.push(`u &= x_m \\pm \\delta = ${texNum(u, d)}`);
        } else if (next.stepKind === 'golden')
          lines.push(
            `u &= x_m \\pm \\rho\\,(\\text{larger segment}) = ${texNum(u, d)} \\quad (\\text{vertex ${vx === null ? 'unusable' : 'rejected'}})`,
          );
        else if (next.stepKind === 'bisect')
          lines.push(
            `f(x_m) &> \\min(f(x_l), f(x_r)) \\;\\Rightarrow\\; u = \\text{midpoint} = ${texNum(u, d)}`,
          );
      }
      break;
    }
    case 'brent': {
      const [a, b] = v.bracket!;
      const xwv = trace[k].info.xwv as number[];
      const u = next?.trials[0]?.[0];
      const vx = next?.vertex ?? null;
      const d = digitsForPoints([...xwv, a, b, u, vx], v.x);
      lines.push(`(x, w, v) &= ${tuple(xwv, d)}`);
      lines.push(`[a,\\, b] &= ${interval(a, b, d)}`);
      if (next && u !== undefined) {
        const tol = nextInfo!.tol as number;
        const x = v.x;
        if (next.stepKind === 'parabolic') {
          if (vx !== null && u === vx) {
            lines.push(`u &= x + p/q = ${texNum(u, d)} \\quad \\text{(parabolic step)}`);
          } else {
            // The vertex is replaced by x ± tol: too close to an end, or a step below tol.
            const nearEnd = vx !== null && (vx - a < 2 * tol || b - vx < 2 * tol);
            lines.push(`x + p/q &= ${texNum(vx, d)}`);
            lines.push(
              `u &= x \\pm \\text{tol} = ${texNum(u, d)} \\quad ${
                nearEnd
                  ? '(x + p/q \\text{ within } 2\\,\\text{tol of } a \\text{ or } b)'
                  : `(|p/q| < \\text{tol} = ${texNum(tol, 3)})`
              }`,
            );
          }
        } else {
          const toB = x < 0.5 * (a + b);
          const e = toB ? b - x : a - x;
          const seg = toB ? 'b - x' : 'a - x';
          const why = vx === null ? 'golden step' : 'golden step; vertex rejected';
          if (Math.abs(RHO * e) >= tol)
            lines.push(`u &= x + \\rho\\,(${seg}) = ${texNum(u, d)} \\quad \\text{(${why})}`);
          else {
            lines.push(
              `\\rho\\,(${seg}) &= ${texNum(RHO * e, 3)} \\quad (|\\cdot| < \\text{tol} = ${texNum(tol, 3)})`,
            );
            lines.push(`u &= x \\pm \\text{tol} = ${texNum(u, d)}`);
          }
        }
      }
      break;
    }
    case 'newton': {
      const g = trace[k].info.g as number;
      const h = trace[k].info.h as number;
      const d = digitsForPoints([v.x, next?.nextX], v.x);
      lines.push(`x_{${k}} &= ${texNum(v.x, d)}`);
      lines.push(`f'(x_{${k}}) &= ${texNum(g, 4)},\\quad f''(x_{${k}}) = ${texNum(h, 4)}`);
      if (next && next.nextX !== null) {
        if (next.stepKind === 'newton')
          lines.push(
            `x_{${k + 1}} &= x_{${k}} - \\frac{f'(x_{${k}})}{f''(x_{${k}})} = ${texNum(next.nextX, d)}`,
          );
        else {
          const alpha = trace[k + 1].info.alpha as number;
          lines.push(
            `f''(x_{${k}}) &\\le 0 \\;\\Rightarrow\\; x_{${k + 1}} = x_{${k}} - \\alpha f'(x_{${k}}),\\ \\alpha = ${texNum(alpha, 4)}`,
          );
        }
      }
      break;
    }
    case 'bracketing': {
      const t = trace[k].info.triple as number[];
      const ft = trace[k].info.f_triple as number[];
      const tr = next?.trials ?? [];
      const d = digitsForPoints([...t, ...tr.map((p) => p[0])], v.x);
      lines.push(`(a, b, c) &= ${tuple(t, d)}`);
      if (next) {
        const [u1, u2] = [tr[0]?.[0], tr[1]?.[0]];
        switch (next.stepKind) {
          case 'golden':
            if (tr.length === 2) {
              // The vertex fell in (b, c) but f(c) ≤ f(u) ≤ f(b): it did not help.
              lines.push(
                `f(b) &> f(c) \\;\\Rightarrow\\; u = \\text{vertex} = ${texNum(u1, d)} \\in (b, c)`,
              );
              lines.push(
                `f(c) &\\le f(u) \\le f(b) \\;\\Rightarrow\\; u' = c + \\varphi\\,(c - b) = ${texNum(u2, d)}`,
              );
            } else
              lines.push(
                `f(b) &> f(c) \\;\\Rightarrow\\; u = c + \\varphi\\,(c - b) = ${texNum(u1, d)}`,
              );
            break;
          case 'limit':
            lines.push(
              `f(b) &> f(c) \\;\\Rightarrow\\; u = u_{\\lim} = b + g\\,(c - b) = ${texNum(u1, d)}`,
            );
            break;
          case 'parabolic':
            lines.push(
              `f(b) &> f(c) \\;\\Rightarrow\\; u = \\text{vertex} = ${texNum(u1, d)} \\in (b, c)`,
            );
            break;
          case 'parabolic_far':
            lines.push(
              `f(b) &> f(c) \\;\\Rightarrow\\; u = \\text{vertex} = ${texNum(u1, d)} \\quad (\\text{beyond } c)`,
            );
            if (tr.length === 2)
              lines.push(
                `f(u) &< f(c) \\;\\Rightarrow\\; u' = u + \\varphi\\,(u - c) = ${texNum(u2, d)}`,
              );
            if (tr.length === 2) lines.push(`(b, c) &\\leftarrow (c, u)`);
            break;
          default:
            if (u1 !== undefined) lines.push(`u &= ${texNum(u1, d)}`);
        }
      } else if (ft && ft[1] <= ft[2] && ft[1] <= ft[0]) {
        lines.push(`f(b) &\\le f(a),\\quad f(b) \\le f(c) \\quad \\text{(a bracket)}`);
      }
      break;
    }
  }
  if (last && kind !== 'bracketing') {
    lines.push(`&\\text{${result.converged ? 'stopping test passed' : 'stopped'} at } k = ${k}`);
  }
  // A conclusion ("⇒ drop [a_k, x₁)", "⇒ u = …") starts its own row, aligned under the "=":
  // the block stays narrow enough for the method card at every width.
  const SPLIT = ' \\;\\Rightarrow\\; ';
  return lines
    .flatMap((l) => {
      const i = l.indexOf(SPLIT);
      return i < 0 ? [l] : [l.slice(0, i), `&\\Rightarrow\\; ${l.slice(i + SPLIT.length)}`];
    })
    .join(' \\\\ ');
}
