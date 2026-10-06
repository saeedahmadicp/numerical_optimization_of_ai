/**
 * The line search of the step under the playhead, seen from the side: φ(α) = f(𝐱ₖ₋₁ + α𝐩) on
 * the step lengths the search tried, the lines of the test the search applies (Armijo's
 * sufficient decrease φ(0) + c₁αφ′(0), Goldstein's lower line φ(0) + (1 − c)αφ′(0), the slope
 * φ′(α) at the accepted α for the Wolfe curvature condition), every trial in order and the
 * accepted α. The caption says which conditions the accepted α meets. Steps that apply no test
 * (pure Newton, pure Barzilai–Borwein, a fixed step) say so instead.
 */
import { useMemo } from 'react';
import type { Problem2D } from '../../core/types';
import { Plot1D, mathMain, mathVar, type Overlay1D } from '../../viz';
import { MathText } from './MathText';
import { fmt, type Ray } from './geometry';
import { tn } from './stepTex';
import styles from './UnconstrainedLab.module.css';

import { SEARCH, type SearchRule } from './search';

export interface RayPanelProps {
  ray: Ray;
  problem: Problem2D;
  slot: number;
  k: number;
  search: SearchRule;
  /** Curvature constant c₂ of the Wolfe searches (0.9, or the CG methods' `c2`). */
  c2: number;
  /** Barzilai–Borwein's nonmonotone reference value (the Armijo line starts there). */
  fRef: number | null;
}

const ok = (b: boolean) => (b ? '\\checkmark' : '\\times');

export function RayPanel({ ray, problem, slot, k, search, c2, fRef }: RayPanelProps) {
  const { origin, dir, trials, alpha } = ray;
  const phi = useMemo(
    () => (a: number) => problem.f([origin[0] + a * dir[0], origin[1] + a * dir[1]]),
    [problem, origin, dir],
  );
  const dphi = (a: number) => {
    const g = problem.grad([origin[0] + a * dir[0], origin[1] + a * dir[1]]);
    return g[0] * dir[0] + g[1] * dir[1];
  };
  const slope = dphi(0);
  const phi0 = phi(0);
  const aMax = Math.max(alpha ?? 0, ...trials.map((t) => t[0]));
  const A = aMax > 0 ? aMax * 1.12 : 1;

  const { lo, hi } = useMemo(() => {
    let lo = phi0;
    for (let i = 0; i <= 160; i++) {
      const y = phi((A * i) / 160);
      if (Number.isFinite(y)) lo = Math.min(lo, y);
    }
    const ref = fRef ?? phi0;
    const span = Math.max(ref - lo, Math.abs(ref) * 1e-9, 1e-12);
    // Trials up to 4 spans above φ(0) are shown at their value; higher ones at the top edge.
    const shown = trials.map((t) => t[1]).filter((y) => Number.isFinite(y) && y <= ref + 4 * span);
    const top = Math.max(ref + 0.9 * span, ...shown.map((y) => y + 0.12 * span));
    return { lo: lo - 0.08 * span, hi: top };
  }, [phi, A, phi0, fRef, trials]);

  const info = SEARCH[search];
  const c1 = info.c1;
  const tested = c1 !== null && Number.isFinite(slope);
  const ref = fRef ?? phi0;
  const overlays: Overlay1D[] = [];
  if (Number.isFinite(slope))
    overlays.push({ kind: 'tangent', x: 0, y: phi0, slope, halfWidth: A * 0.18 });
  if (tested) {
    overlays.push({
      kind: 'segment',
      x1: 0,
      y1: ref,
      x2: A,
      y2: ref + c1 * A * slope,
      dashed: true,
      width: 1.2,
    });
    // Goldstein's lower line: the step must not be too short.
    if (search === 'goldstein')
      overlays.push({
        kind: 'segment',
        x1: 0,
        y1: phi0,
        x2: A,
        y2: phi0 + (1 - c1) * A * slope,
        dashed: true,
        width: 1,
      });
  }
  trials.forEach(([a, f], j) => {
    if (alpha !== null && a === alpha && j === trials.length - 1) return;
    const clipped = !(f <= hi);
    overlays.push({
      kind: 'point',
      x: a,
      y: clipped ? hi - 0.06 * (hi - lo) : f,
      slot,
      shape: 'ring',
      label: clipped ? `${j + 1} ↑` : String(j + 1),
      labelSide: 'left',
    });
  });
  const wolfe = search === 'strong_wolfe' || search === 'weak_wolfe';
  const phiA = alpha !== null ? phi(alpha) : NaN;
  const dphiA = alpha !== null && wolfe ? dphi(alpha) : NaN;
  if (alpha !== null) {
    overlays.push({ kind: 'vline', x: alpha, slot, dashed: true });
    // The curvature condition is about the slope at α: drawn as the tangent there.
    if (wolfe && Number.isFinite(dphiA))
      overlays.push({
        kind: 'tangent',
        x: alpha,
        y: phiA,
        slope: dphiA,
        halfWidth: A * 0.14,
        slot,
      });
    overlays.push({
      kind: 'point',
      x: alpha,
      y: Math.min(phiA, hi),
      slot,
      label: [mathVar('α'), mathMain(` = ${fmt(alpha)}`)],
      // Near the right edge the label goes to the left of the point, inside the plot.
      labelSide: alpha > 0.7 * A ? 'left' : 'right',
    });
  }

  // Which conditions the accepted α meets (the tests of numopt.line_search, N&W §3.1).
  const conditions: string[] = [];
  if (alpha !== null && tested && Number.isFinite(phiA)) {
    const armijo = phiA - ref <= c1 * alpha * slope;
    conditions.push(
      `${search === 'gll' ? 'the GLL test' : 'sufficient decrease'} $\\varphi(\\alpha) \\le ${fRef !== null ? 'f_{\\mathrm{ref}}' : '\\varphi(0)'} + c_1\\alpha\\varphi'(0)$ $${ok(armijo)}$`,
    );
    if (search === 'goldstein') {
      const lower = phiA - phi0 >= (1 - c1) * alpha * slope;
      conditions.push(
        `the lower Goldstein line $\\varphi(\\alpha) \\ge \\varphi(0) + (1 - c)\\alpha\\varphi'(0)$ $${ok(lower)}$`,
      );
    }
    if (wolfe && Number.isFinite(dphiA)) {
      const curv = search === 'strong_wolfe' ? Math.abs(dphiA) <= -c2 * slope : dphiA >= c2 * slope;
      conditions.push(
        search === 'strong_wolfe'
          ? `strong curvature $|\\varphi'(\\alpha)| = ${tn(Math.abs(dphiA), 3)} \\le c_2|\\varphi'(0)| = ${tn(-c2 * slope, 3)}$ $${ok(curv)}$`
          : `curvature $\\varphi'(\\alpha) = ${tn(dphiA, 3)} \\ge c_2\\varphi'(0) = ${tn(c2 * slope, 3)}$ $${ok(curv)}$`,
      );
    }
  }

  const n = trials.length === 1 ? 'one step length' : `${trials.length} step lengths`;
  let caption: string;
  switch (search) {
    case 'newton':
      caption = 'No line search: pure Newton takes the full step $\\alpha = 1$.';
      break;
    case 'bb':
      caption = `No line search (nonmonotone off): the Barzilai–Borwein step${alpha !== null ? ` $\\alpha_{${k}} = ${tn(alpha)}$` : ''} is taken without a test.`;
      break;
    case 'fixed':
      caption = `No line search: the fixed step${alpha !== null ? ` $\\alpha = ${tn(alpha)}$` : ''} is taken without a test.`;
      break;
    case 'schedule':
      caption = `No line search: the schedule's step${alpha !== null ? ` $\\alpha_{${k}} = h/L = ${tn(alpha)}$` : ''} is fixed before the run and taken without a test. A long step can end past the minimum along the ray.`;
      break;
    case 'fista':
      caption =
        `Beck–Teboulle backtracking from $\\mathbf{y}_{${k}}$ along $-\\nabla f(\\mathbf{y}_{${k}})$ tried ${n}, $\\alpha = 1/\\bar L$,` +
        (alpha !== null
          ? ` and accepted $\\alpha_{${k}} = 1/L_{${k}} = ${tn(alpha)}$.`
          : ' and found no acceptable step.');
      break;
    case 'exact_quadratic':
      caption = `The exact minimizer of the quadratic model along $\\mathbf{p}$${alpha !== null ? `: $\\alpha_{${k}} = ${tn(alpha)}$` : ''}.`;
      break;
    default:
      caption =
        `The ${info.name} tried ${n}` +
        (alpha !== null
          ? ` and accepted $\\alpha_{${k}} = ${tn(alpha)}$.`
          : ' and found no acceptable step.');
  }
  caption +=
    ` The slope $\\varphi'(0) = \\nabla f^{\\top}\\mathbf{p} = ${tn(slope, 3)}$` +
    (slope < 0 ? ' points downhill.' : ' is not negative: the direction goes uphill.');
  if (tested)
    caption +=
      search === 'goldstein'
        ? ` Dashed: the Goldstein lines with $c = ${tn(c1)}$ (upper: sufficient decrease; lower: not too short).`
        : ` Dashed: the sufficient-decrease line, $c_1 = ${tn(c1)}$${fRef !== null ? ', from the nonmonotone reference $f_{\\mathrm{ref}}$' : ''}.`;
  if (wolfe) caption += ` Solid at $\\alpha$: the slope $\\varphi'(\\alpha)$, $c_2 = ${tn(c2)}$.`;
  if (conditions.length) caption += ` The accepted $\\alpha$ meets: ${conditions.join('; ')}.`;

  return (
    <div className={styles.rayPanel}>
      <div className={styles.rayPlot}>
        <Plot1D
          f={phi}
          domain={[0, A]}
          yDomain={[lo, hi]}
          overlays={overlays}
          xLabel="α"
          yName={[mathVar('φ'), mathMain('('), mathVar('α'), mathMain(')')]}
          ariaLabel={`φ(α) = f(x_${k - 1} + α p) along the search direction of step ${k}, with ${trials.length} trial step lengths${alpha !== null ? ` and the accepted α = ${alpha.toPrecision(3)}` : ''}.`}
        />
      </div>
      <p className={styles.rayCaption}>
        <MathText text={caption} />
      </p>
    </div>
  );
}
