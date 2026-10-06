/**
 * The unconstrained lab's step geometry and step text, checked against the traces of the real TS
 * ports: every method on several problems at every step (no throw, the right marks), and the
 * identities each drawing claims (tip-to-tail sums, ellipses through 𝐱ₖ₋₁, centers at the
 * Newton point, the trust-region disk).
 */
import { describe, expect, it } from 'vitest';
import './setup';
import { defaults, getMethod, listMethods } from '../../core/registry';
import { getProblem } from '../../problems/registry';
import type { Problem2D, Result } from '../../core/types';
import type { Overlay2D } from '../../viz/overlays2d';
import { GROUPS, KNOWN_METHODS, LAB_DOCS, RAY_KINDS, kindOf, withLabDoc } from './catalog';
import {
  buildGeometry,
  inv2,
  isSpd,
  nearestMinimizer,
  rayOf,
  seriesValues,
  stepAt,
  type Geometry,
} from './geometry';
import { stepView, stripOf, tn, weightsOf } from './stepTex';
import { columnsFor } from './columns';
import { curvatureOnView, needsConstant, niceUp } from './constants';
import { STRIP_WINDOW, barScale, stripWindow } from './strip';

const run = (id: string, pid: string, extra: Record<string, unknown> = {}) => {
  const m = getMethod<Problem2D>(id);
  const problem = getProblem<Problem2D>(pid);
  const params = { ...defaults(m.spec), ...extra };
  const result = m.fn(problem, params as never) as Result;
  return { problem, params, result };
};

const geo = (id: string, pid: string, g: number, extra: Record<string, unknown> = {}) => {
  const { problem, params, result } = run(id, pid, extra);
  return {
    problem,
    result,
    params,
    geometry: buildGeometry({
      kind: kindOf(id)!,
      method: id,
      trace: result.trace,
      g,
      params,
      problem,
      slot: 0,
      labels: true,
      span: 4,
    }),
  };
};

const of = <K extends Overlay2D['kind']>(g: Geometry, kind: K) =>
  g.overlays.filter((o): o is Extract<Overlay2D, { kind: K }> => o.kind === kind);

const arr = (x: unknown) => x as number[];
const close = (a: readonly number[], b: readonly number[], tol = 1e-9) =>
  a.forEach((v, i) =>
    expect(Math.abs(v - b[i])).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i]))),
  );

describe('catalog', () => {
  const ids = listMethods('unconstrained').map((m) => m.spec.id);
  it('groups and draws every registered unconstrained method exactly once', () => {
    const grouped = GROUPS.flatMap((g) => g.methods);
    expect(new Set(grouped).size).toBe(grouped.length);
    expect([...grouped].sort()).toEqual([...ids].sort());
    expect([...KNOWN_METHODS].sort()).toEqual([...ids].sort());
    expect(ids.length).toBe(42);
  });
  it('gives every method a rule in bold-vector notation and a short rate', () => {
    for (const id of ids) {
      const doc = withLabDoc(getMethod(id)).doc!;
      expect(LAB_DOCS[id], id).toBeDefined();
      expect(doc.rule.length).toBeGreaterThan(10);
      expect(doc.order!.length).toBeLessThanOrEqual(20);
      expect(doc.intuition.length).toBeGreaterThan(20);
      expect(doc.cons?.length, `${id} names when it fails`).toBeGreaterThan(0);
      expect(doc.quantities).toEqual([]);
    }
  });
});

describe('stepAt', () => {
  it('shows the step being traversed while playing and the step that led here when paused', () => {
    expect(stepAt(0, 10)).toEqual({ g: 0, u: 1 });
    expect(stepAt(3, 10)).toEqual({ g: 3, u: 1 });
    expect(stepAt(3.25, 10).g).toBe(4);
    expect(stepAt(3.25, 10).u).toBeCloseTo(0.25);
    expect(stepAt(42, 10).g).toBe(9);
    expect(stepAt(5, 1)).toEqual({ g: 0, u: 1 });
  });
});

/**
 * The constants a certified schedule or OGM needs on a problem that is not a quadratic: L (and μ)
 * from the curvature on the plotted region, as the lab's "Use L = …" button sets them; null when
 * the method cannot run there (the strongly convex schedule on a nonconvex region).
 */
function constantsFor(id: string, pid: string): Record<string, number> | null {
  const m = getMethod(id);
  if (!m.spec.params.some((p) => p.name === 'L')) return {};
  const problem = getProblem<Problem2D>(pid);
  if (problem.tags?.includes('quadratic')) return {};
  const c = curvatureOnView(problem)!;
  if (id === 'silver_gd_strongly_convex')
    return c.muMin > 0 ? { L: niceUp(c.lMax), mu: Number(c.muMin.toPrecision(2)) } : null;
  return { L: niceUp(c.lMax) };
}

describe('every method, every step', () => {
  const problems = ['rosenbrock', 'himmelblau', 'quadratic_ill', 'six_hump_camel'];
  for (const id of listMethods('unconstrained').map((m) => m.spec.id)) {
    it(`${id}: geometry, step text and columns at every step`, () => {
      for (const pid of problems) {
        const constants = constantsFor(id, pid);
        if (constants === null) {
          // The lab shows ConstantHint instead of a run; the error is the one it recognizes.
          let error = '';
          try {
            run(id, pid, { max_iter: 60 });
          } catch (e) {
            error = (e as Error).message;
          }
          expect(needsConstant(error), `${id} on ${pid}: ${error}`).toBe(true);
          continue;
        }
        const { problem, params, result } = run(id, pid, { max_iter: 60, ...constants });
        const kind = kindOf(id)!;
        let drawn = 0;
        result.trace.forEach((_, g) => {
          const input = { kind, method: id, trace: result.trace, g, params, problem };
          const geometry = buildGeometry({ ...input, slot: 1, labels: true, span: 4 });
          if (geometry.overlays.length + geometry.curves.length > 0) drawn++;
          for (const o of geometry.overlays)
            for (const v of Object.values(o))
              if (typeof v === 'number') expect(Number.isFinite(v), `${id} ${pid} ${g}`).toBe(true);
          const view = stepView(input);
          expect(view.quantities.length).toBeGreaterThan(0);
          if (g > 0) expect(view.note.length, `${id} ${pid} ${g}`).toBeGreaterThan(0);
          // Inline math is balanced, so every $…$ segment is typeset.
          expect((view.note.match(/(?<!\\)\$/g) ?? []).length % 2, view.note).toBe(0);
          for (const c of columnsFor(kind)) c.value(result.trace[g]);
          if (RAY_KINDS.has(kind) && g > 0) {
            const ray = rayOf(kind, result.trace, g);
            if (ray) expect(Math.hypot(...ray.dir)).toBeGreaterThan(0);
          }
        });
        // Something is drawn for most steps (rejected or skipped steps may draw little).
        if (result.trace.length > 2)
          expect(drawn, `${id} on ${pid}`).toBeGreaterThanOrEqual(
            Math.min(2, result.trace.length - 1),
          );
      }
    });
  }
});

describe('the drawings are exact', () => {
  it('heavy ball: βv_{k−1} then −α∇f(x_{k−1}) end at x_k', () => {
    const { geometry, result, problem, params } = geo('momentum', 'quadratic_ill', 5);
    const [mom, grad] = of(geometry, 'arrow');
    const x0 = arr(result.trace[4].x);
    const x1 = arr(result.trace[5].x);
    close(arr(mom.from), x0);
    close(arr(grad.to), x1);
    const g = problem.grad(x0);
    close(
      [grad.to[0] - grad.from[0], grad.to[1] - grad.from[1]],
      [-(params.lr as number) * g[0], -(params.lr as number) * g[1]],
    );
  });

  it('Nesterov: the gradient leg starts at the look-ahead point and is −α∇f there', () => {
    const { geometry, result, problem, params } = geo('nesterov', 'rosenbrock', 7);
    const legs = of(geometry, 'arrow');
    const la = arr(result.trace[7].info.lookahead);
    close(arr(legs[0].to), la);
    const g = problem.grad(la);
    close(
      [legs[1].to[0] - legs[1].from[0], legs[1].to[1] - legs[1].from[1]],
      [-(params.lr as number) * g[0], -(params.lr as number) * g[1]],
    );
  });

  it('Adam: the step −D⊙m̂ lands on the drawn ellipse and reaches x_k', () => {
    const { geometry, result } = geo('adam', 'quadratic_bowl', 9);
    const [e] = of(geometry, 'ellipse');
    const x0 = arr(result.trace[8].x);
    const x1 = arr(result.trace[9].x);
    const d = [x1[0] - x0[0], x1[1] - x0[1]];
    const q = e.matrix[0][0] * d[0] ** 2 + e.matrix[1][1] * d[1] ** 2;
    expect(q).toBeCloseTo(1, 9);
    close(arr(e.center), x0);
  });

  it('Newton: the model level set is centered at the Newton point x_{k−1} + p = x_k', () => {
    const { geometry, result } = geo('pure_newton', 'rosenbrock', 4);
    const [e] = of(geometry, 'ellipse');
    close(arr(e.center), arr(result.trace[4].x), 1e-9);
    // …and passes through x_{k−1}.
    const x0 = arr(result.trace[3].x);
    const d = [x0[0] - e.center[0], x0[1] - e.center[1]];
    const M = e.matrix;
    const q = d[0] * (M[0][0] * d[0] + M[0][1] * d[1]) + d[1] * (M[1][0] * d[0] + M[1][1] * d[1]);
    expect(q / (e.radius ?? 1) ** 2).toBeCloseTo(1, 9);
  });

  it('BFGS: the model ellipse uses B = H⁻¹, passes through x_{k−1} and is centered at x_{k−1} + p', () => {
    const { geometry, result } = geo('bfgs', 'rosenbrock', 6);
    const ellipses = of(geometry, 'ellipse');
    const qn = ellipses.find((e) => e.slot === 0)!;
    const H = result.trace[5].info.H as number[][];
    close(qn.matrix.flat(), inv2(H)!.flat(), 1e-12);
    const x0 = arr(result.trace[5].x);
    const p = arr(result.trace[6].info.direction);
    close(arr(qn.center), [x0[0] + p[0], x0[1] + p[1]], 1e-9);
  });

  it('BFGS on a quadratic: the dashed ink ellipse is the true Hessian model (∇²f ≻ 0)', () => {
    const { geometry, problem, result } = geo('bfgs', 'quadratic_ill', 2);
    const truth = of(geometry, 'ellipse').find((e) => e.slot === undefined)!;
    expect(truth.dashed).toBe(true);
    close(truth.matrix.flat(), problem.hess!(arr(result.trace[1].x)).flat(), 1e-12);
    // Its center is the exact minimizer: Newton's model of a quadratic is the quadratic.
    close(arr(truth.center), [0, 0], 1e-9);
  });

  it('CG: −α∇f then αβ_{k−1}d_{k−2} end at x_k (no restart)', () => {
    const { result, problem, params } = run('cg_polak_ribiere', 'rosenbrock');
    const g = result.trace.findIndex(
      (_s, i) => i >= 2 && result.trace[i - 1].info.restart === null,
    );
    const geometry = buildGeometry({
      kind: 'cg',
      method: 'cg_polak_ribiere',
      trace: result.trace,
      g,
      params,
      problem,
      slot: 0,
      labels: true,
      span: 4,
    });
    const [lead, tail] = of(geometry, 'arrow');
    close(arr(lead.from), arr(result.trace[g - 1].x));
    close(arr(tail.to), arr(result.trace[g].x), 1e-8);
  });

  it('trust region: the disk is the model region of the step, with Cauchy and Newton points', () => {
    const { geometry, result } = geo('trust_region_dogleg', 'rosenbrock', 2);
    const [disk] = of(geometry, 'disk');
    close(arr(disk.center), arr(result.trace[2].info.center));
    expect(disk.radius).toBe(result.trace[2].info.radius);
    // Step 2 is rejected (ρ < 0): the trial point is marked and the radius shrinks to Δ/4.
    expect(result.trace[2].info.accepted).toBe(false);
    expect(of(geometry, 'disk').some((d) => d.dashed && d.radius === 0.25)).toBe(true);
    expect(of(geometry, 'polyline').length).toBeGreaterThan(0); // the dogleg path
  });

  it('Nelder–Mead: the simplex after the operation, the previous one dashed', () => {
    const { geometry, result } = geo('nelder_mead', 'rosenbrock', 3);
    const [old, cur] = of(geometry, 'polygon');
    expect(old.dashed).toBe(true);
    expect(cur.points).toEqual(result.trace[3].info.simplex);
  });

  it('a schedule step: the arrow is x_{k−1} → x_k and the ring marks where the step 1/L ends', () => {
    const { geometry, result, problem } = geo('silver_gd', 'quadratic_ill', 4);
    const [arrow] = of(geometry, 'arrow');
    close(arr(arrow.from), arr(result.trace[3].x));
    close(arr(arrow.to), arr(result.trace[4].x));
    const L = result.extra.L as number;
    const g = problem.grad(arr(result.trace[3].x));
    const ring = of(geometry, 'point').find((p) => p.shape === 'ring')!;
    close(arr(ring.at), [
      arr(result.trace[3].x)[0] - g[0] / L,
      arr(result.trace[3].x)[1] - g[1] / L,
    ]);
    // Certified checkpoints so far (k = 1, 3) are dots on the path.
    expect(of(geometry, 'point').filter((p) => p.shape === 'dot').length).toBe(2);
  });

  it('FISTA: the gradient leg is −∇f(y_k)/L_k from y_k; every restart so far is ringed', () => {
    const { result, problem, params } = run('fista', 'quadratic_ill');
    const restarts = result.extra.restarts as number[];
    const g = restarts[1] + 1;
    const geometry = buildGeometry({
      kind: 'fista',
      method: 'fista',
      trace: result.trace,
      g,
      params,
      problem,
      slot: 0,
      labels: true,
      span: 4,
    });
    const legs = of(geometry, 'arrow');
    const leg = legs[legs.length - 1];
    const y = arr(result.trace[g].info.y);
    const gy = problem.grad(y);
    const L = result.trace[g].info.L as number;
    close(arr(leg.from), y);
    close(arr(leg.to), [y[0] - gy[0] / L, y[1] - gy[1] / L], 1e-9);
    const rings = of(geometry, 'point').filter((p) => p.shape === 'ring' && p.radius === 5);
    expect(rings.length).toBe(2);
    close(arr(rings[1].at), arr(result.trace[restarts[1]].x));
  });

  it('Anderson: x̄ = Σ cᵢxᵢ over the drawn history, and the step ends at x_k', () => {
    const { geometry, result } = geo('anderson_gd', 'himmelblau', 6, { lr: 0.01, x0: [1, 1] });
    const info = result.trace[6].info;
    const hist = info.history as number[][];
    const c = info.coefficients as number[];
    const xb = [0, 1].map((d) => hist.reduce((s, h, j) => s + c[j] * h[d], 0));
    const cross = of(geometry, 'point').find((p) => p.shape === 'cross')!;
    close(arr(cross.at), xb, 1e-9);
    expect(of(geometry, 'point').filter((p) => p.shape === 'ring').length).toBe(hist.length);
    const step = of(geometry, 'arrow').find((a) => a.slot === 0)!;
    close(arr(step.to), arr(result.trace[6].x));
  });

  it('ARC: the ball has radius ‖s‖ = λ/σ and the trial point is on it', () => {
    const { geometry, result } = geo('arc', 'rosenbrock', 1);
    const info = result.trace[1].info;
    const [disk] = of(geometry, 'disk');
    close(arr(disk.center), arr(info.center));
    close([disk.radius], [(info.lambda as number) / (info.sigma as number)], 1e-9);
    const t = arr(info.trial_point);
    close([Math.hypot(t[0] - disk.center[0], t[1] - disk.center[1])], [disk.radius], 1e-9);
    expect(geometry.curves[0].segs.length).toBeGreaterThan(0); // the cubic model's level set
  });

  it('regularized Newton: the model with ∇²f + λI through x_{k−1} is centered at x_k', () => {
    const { geometry, result } = geo('reg_newton', 'rosenbrock', 3);
    const e = of(geometry, 'ellipse')[0];
    close(arr(e.center), arr(result.trace[3].x), 1e-9);
    const lam = result.trace[3].info.lambda as number;
    const H = result.trace[2].info.hess as number[][];
    close([e.matrix[0][0], e.matrix[1][1]], [H[0][0] + lam, H[1][1] + lam], 1e-12);
  });

  it('every symmetric 2×2 inverse is exact', () => {
    const M = [
      [4, 1],
      [1, 3],
    ];
    const I = inv2(M)!;
    close([4 * I[0][0] + I[1][0], 4 * I[0][1] + I[1][1]], [1, 0], 1e-15);
    expect(isSpd(M)).toBe(true);
    expect(
      isSpd([
        [1, 2],
        [2, 1],
      ]),
    ).toBe(false);
  });
});

describe('convergence measures', () => {
  it('measure against the minimizer nearest the last iterate', () => {
    const { problem, result } = run('modified_newton', 'himmelblau');
    const star = nearestMinimizer(problem, result.x)!;
    close(arr(star), [3.5844283403304917, -1.8481265269644036]);
    const gap = seriesValues('gap', result.trace, problem);
    expect(gap[0]).toBeCloseTo(problem.f(arr(result.trace[0].x)) - problem.f(arr(star)), 9);
    expect(Math.min(...(gap as number[]))).toBeGreaterThan(0);
  });
  it('derivative-free runs get ‖∇f‖ evaluated by the lab', () => {
    const { problem, result } = run('nelder_mead', 'rosenbrock');
    const g = seriesValues('grad', result.trace, problem);
    expect(result.trace[0].gradNorm).toBeNull();
    expect(g[0]).toBeCloseTo(Math.hypot(...problem.grad(arr(result.trace[0].x))), 12);
  });
});

describe('number typesetting', () => {
  it('uses ×10ⁿ outside [10⁻³, 10⁵) and plain digits inside', () => {
    expect(tn(0.0123)).toBe('0.0123');
    expect(tn(1.5e-8)).toBe('1.5\\times 10^{-8}');
    expect(tn(1e-8)).toBe('10^{-8}');
    expect(tn(-2.5e6)).toBe('-2.5\\times 10^{6}');
    expect(tn(null)).toBe('\\text{—}');
  });
});

describe('step strips and constants', () => {
  it('a schedule strip has one bar per step, the h = 2 line and the certified checkpoints', () => {
    const { result } = run('silver_gd', 'quadratic_ill', { max_iter: 100 });
    const strip = stripOf({ kind: 'schedule', trace: result.trace })!;
    expect(strip.kind).toBe('bars');
    if (strip.kind !== 'bars') return;
    expect(strip.values.length).toBe(result.trace.length);
    expect(strip.values[0]).toBeNull();
    expect(strip.marks).toEqual([1, 3, 7, 15, 31, 63]);
    const scale = barScale(strip);
    expect(scale(Math.SQRT2)).toBeLessThan(scale(2));
    expect(scale(Math.max(...(strip.values.slice(1) as number[])))).toBeCloseTo(1, 12);
  });

  it('FISTA marks its restarts, ARC its rejected steps', () => {
    const f = run('fista', 'quadratic_ill').result;
    const s = stripOf({ kind: 'fista', trace: f.trace })!;
    expect(s.kind === 'bars' && s.marks).toEqual(f.extra.restarts);
    const a = run('arc', 'rosenbrock').result;
    const t = stripOf({ kind: 'arc', trace: a.trace })!;
    expect(t.kind === 'bars' && t.marks.length).toBe(a.extra.n_rejected);
  });

  it('the strip window pages forward with the playhead and keeps 64 steps', () => {
    expect(stripWindow(30, 7)).toEqual([1, 30]);
    expect(stripWindow(255, 0)).toEqual([1, STRIP_WINDOW]);
    expect(stripWindow(255, 64)).toEqual([1, 64]);
    expect(stripWindow(255, 65)).toEqual([65, 128]);
    expect(stripWindow(255, 255)).toEqual([193, 255]);
  });

  it('AA weights: empty for a gradient step, else the window’s c* (they sum to 1)', () => {
    const { result } = run('anderson_gd', 'himmelblau', { lr: 0.01, x0: [1, 1] });
    expect(weightsOf(result.trace, 1).weights).toEqual([]);
    const w = weightsOf(result.trace, 5);
    expect(w.weights.length).toBe((result.trace[5].info.memory as number) + 1);
    expect(w.weights.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12);
    expect(w.names[w.names.length - 1]).toBe('\\mathbf{x}_{4}');
  });

  it('constants: L from the curvature on the view, rounded up; the error is recognized', () => {
    expect(niceUp(2431.7)).toBe(2500);
    expect(niceUp(0.0123)).toBe(0.013);
    expect(niceUp(50)).toBe(50);
    const q = curvatureOnView(getProblem<Problem2D>('quadratic_ill'))!;
    expect(q.lMax).toBeCloseTo(50, 9);
    expect(q.muMin).toBeCloseTo(1, 9);
    let error = '';
    try {
      run('silver_gd', 'rosenbrock');
    } catch (e) {
      error = (e as Error).message;
    }
    expect(needsConstant(error)).toBe(true);
    expect(needsConstant('μ = 1 exceeds L = 0.5')).toBe(false);
  });
});
