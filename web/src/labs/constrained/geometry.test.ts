/**
 * Constrained lab: the drawn mathematics is checked against the methods' own numbers.
 *
 *   - each merit landscape, evaluated at x_k, equals the method's `info.merit` at step k;
 *   - at a KKT point the chain λ_i∇c_i ends at the tip of −∇f;
 *   - the claims of the default view and of the "Try this" notes hold;
 *   - every TeX string the lab builds (rules with numbers, landscapes) parses in KaTeX.
 */
import { describe, expect, it } from 'vitest';
import katex from 'katex';
import { getMethod } from '../../core/registry';
import { getProblem } from '../../problems/registry';
import type { Result, Step } from '../../core/types';
import type { ConstrainedProblem } from '../../problems/constrained';
import './setup';
import { DEFAULT_PROBLEM, DEFAULT_SELECTION, PRESETS } from './config';
import {
  atoms,
  centralPath,
  clipSegment,
  minimizerLabelSide,
  nearestMinimizer,
  feasibleOracle,
  feasibleSet,
  geometryIndex,
  kktArrows,
  linearization,
  linearizedConstraints,
  meritLandscape,
  stepGeometry,
  texNum,
} from './geometry';
import { VIEW_DOMAIN } from './config';
import { compactNum, constraintStatus, filledRule, ruleInstance, twoLineLatex } from './rules';
import { listProblems } from '../../problems/registry';

function run(id: string, pid: string, params: Record<string, unknown> = {}): Result {
  const { spec, fn } = getMethod(id);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(pid), { ...defaults, ...(params as Record<string, never>) });
}
const P = (id: string) => getProblem<ConstrainedProblem>(id);
const parses = (tex: string) =>
  expect(() => katex.renderToString(tex, { throwOnError: true, displayMode: true })).not.toThrow();

describe('merit landscapes are the functions the methods minimize', () => {
  const cases: [string, string][] = [
    ['quadratic_penalty', 'quadratic_disk'],
    ['quadratic_penalty', 'circle_eq'],
    ['augmented_lagrangian', 'circle_eq'],
    ['augmented_lagrangian', 'hs21'],
    ['log_barrier', 'hs21'],
    ['log_barrier', 'rosenbrock_unit_disk'],
    ['sqp', 'rosenbrock_unit_disk'],
    ['sqp', 'linear_eq_quadratic'],
  ];
  for (const [m, pid] of cases)
    it(`${m} on ${pid}: landscape(x_k) = info.merit`, () => {
      const p = P(pid);
      const r = run(m, pid);
      for (const s of r.trace) {
        const L = meritLandscape(m, s, p)!;
        const x = s.x as number[];
        const want = s.info.merit as number;
        const got = m === 'log_barrier' ? L.f(x[0], x[1]) * (s.info.t as number) : L.f(x[0], x[1]);
        expect(Math.abs(got - want)).toBeLessThanOrEqual(1e-9 * (1 + Math.abs(want)));
        parses(L.tex);
      }
    });
  it('has no landscape for the projection methods', () => {
    const s = run('frank_wolfe', 'quadratic_disk').trace[1];
    expect(meritLandscape('frank_wolfe', s, P('quadratic_disk'))).toBeNull();
  });
  it('the barrier landscape is undefined outside the strict interior', () => {
    const s = run('log_barrier', 'quadratic_disk').trace[3];
    const L = meritLandscape('log_barrier', s, P('quadratic_disk'))!;
    expect(L.f(2, 1)).toBeNaN();
    expect(Number.isFinite(L.f(0, 0))).toBe(true);
  });
});

describe('KKT picture', () => {
  it('closes at the solution: the chain λᵢ∇cᵢ ends at the tip of −∇f (two active constraints)', () => {
    const p = P('halfplanes_quadratic');
    const r = run('sqp', 'halfplanes_quadratic');
    const last = r.trace[r.trace.length - 1];
    const pic = kktArrows(last, p, 0, 5)!;
    expect(pic.terms).toEqual([0, 1]);
    const arrows = pic.overlays.filter((o) => o.kind === 'arrow') as unknown as {
      from: readonly number[];
      to: readonly number[];
    }[];
    const tipF = arrows[arrows.length - 1].to; // −∇f is drawn last, on top of the chain
    const chainEnd = arrows[arrows.length - 2].to;
    expect(Math.hypot(tipF[0] - chainEnd[0], tipF[1] - chainEnd[1])).toBeLessThan(1e-9);
  });
  it('the equality multiplier of circle_eq converges to ν⋆ = 1/√2', () => {
    const r = run('augmented_lagrangian', 'circle_eq');
    const lam = r.extra.multipliers as number[];
    expect(lam[0]).toBeCloseTo(Math.SQRT1_2, 7);
  });
  it('shows −∇f alone for methods without multipliers', () => {
    const r = run('projected_gradient', 'quadratic_disk');
    const pic = kktArrows(r.trace[1], P('quadratic_disk'), 0, 4)!;
    expect(pic.terms).toEqual([]);
    expect(pic.overlays.filter((o) => o.kind === 'arrow')).toHaveLength(1);
  });
});

describe('step geometry', () => {
  it('projected gradient: the arc starts at x_{k−1} and the projection ends at x_k', () => {
    const p = P('quadratic_disk');
    const r = run('projected_gradient', 'quadratic_disk');
    const g = stepGeometry('projected_gradient', r.trace, 1, 0, p, feasibleOracle(p));
    const arc = g.find((o) => o.kind === 'polyline') as { points: readonly (readonly number[])[] };
    expect(arc.points[0]).toEqual(r.trace[0].x);
    const proj = g.find((o) => o.kind === 'segment') as { to: readonly number[] };
    expect(proj.to).toEqual(r.trace[1].x);
    // The projection is the nearest point of the disk: x_k = u/‖u‖ when u is outside.
    const u = r.trace[1].info.unprojected as number[];
    const nu = Math.hypot(u[0], u[1]);
    if (nu > 1) expect((r.trace[1].x as number[])[0]).toBeCloseTo(u[0] / nu, 12);
  });
  it('Frank–Wolfe: atoms are LMO vertices on the circle', () => {
    const r = run('frank_wolfe', 'quadratic_disk');
    const a = atoms(r.trace, 40);
    expect(a.length).toBeGreaterThan(2);
    for (const [x, y] of a) expect(Math.hypot(x, y)).toBeCloseTo(1, 12);
  });
  it('log barrier: the central path is the sequence of stage end points, all strictly inside', () => {
    const p = P('quadratic_disk');
    const r = run('log_barrier', 'quadratic_disk', { t0: 0.1, mu: 4 });
    const path = centralPath(r.trace, r.trace.length - 1);
    const stages = (r.trace[r.trace.length - 1].info.outer as number) + 1;
    expect(path.length).toBe(stages);
    for (const q of path) expect(p.constraints[0].fun(q)).toBeLessThan(0);
    // x⋆(t) moves monotonically toward the circle.
    const radii = path.map(([x, y]) => Math.hypot(x, y));
    for (let i = 1; i < radii.length; i++) expect(radii[i]).toBeGreaterThan(radii[i - 1] - 1e-12);
  });
  it('SQP: the working-set linearizations pass through x_{k−1} + p', () => {
    const p = P('rosenbrock_unit_disk');
    const r = run('sqp', 'rosenbrock_unit_disk');
    for (let k = 1; k < r.trace.length; k++) {
      const s = r.trace[k];
      const from = s.info.from as number[];
      const d = s.info.direction as number[];
      const c0 = r.trace[k - 1].info.constraints as number[];
      for (const i of s.info.working_set as number[]) {
        const a = p.constraints[i].grad(from);
        const seg = linearization(a, c0[i], from, p.domain)!;
        // The QP step satisfies the linearized active constraint: q = x_{k−1} + p lies on the segment.
        const q = [from[0] + d[0], from[1] + d[1]];
        const [A, B] = seg;
        const cross = (B[0] - A[0]) * (q[1] - A[1]) - (B[1] - A[1]) * (q[0] - A[0]);
        expect(Math.abs(cross) / Math.hypot(B[0] - A[0], B[1] - A[1])).toBeLessThan(1e-9);
      }
    }
    expect(linearizedConstraints(r.trace, 3, p).length).toBe(p.constraints.length);
  });
  it('geometryIndex shows the move into x_k while the head travels', () => {
    expect(geometryIndex(0, 5)).toBe(0);
    expect(geometryIndex(0.3, 5)).toBe(1);
    expect(geometryIndex(1, 5)).toBe(1);
    expect(geometryIndex(9, 5)).toBe(4);
  });
  it('every problem has a labelled boundary per constraint', () => {
    for (const id of ['hs21', 'box_quadratic', 'circle_eq', 'mishra_bird_constrained']) {
      const p = P(id);
      const o = feasibleSet(p);
      expect(o.filter((x) => x.kind === 'implicit')).toHaveLength(p.constraints.length);
      // Equalities have no interior: nothing is hatched for circle_eq.
      expect(o.some((x) => x.kind === 'constraints')).toBe(id !== 'circle_eq');
    }
  });
});

describe('claims of the default view and the presets', () => {
  it('default: SQP leaves the disk and returns in 10 iterations; the barrier stays inside; the penalty approaches from outside', () => {
    const p = P(DEFAULT_PROBLEM);
    const byId = Object.fromEntries(
      DEFAULT_SELECTION.map((s) => [s.id, run(s.id, p.id, s.params)]),
    );
    const c = (s: Step) => p.constraints[0].fun(s.x as number[]);
    expect(byId.sqp.nIter).toBe(10);
    expect(byId.sqp.trace.some((s) => c(s) > 0)).toBe(true);
    expect(byId.log_barrier.trace.every((s) => c(s) < 0)).toBe(true);
    const pen = byId.quadratic_penalty.trace;
    expect(pen.slice(-5).every((s) => c(s) > 0)).toBe(true);
    for (const r of Object.values(byId)) expect(r.converged).toBe(true);
  });
  it('Frank–Wolfe needs 625 iterations where projected gradient needs 11', () => {
    const pr = PRESETS.find((x) => x.id === 'fw-sublinear')!;
    const [fw, pg] = pr.methods!.map((m) => run(m.id, pr.problem!, m.params));
    expect([fw.nIter, fw.converged]).toEqual([625, true]);
    expect([pg.nIter, pg.converged]).toEqual([11, true]);
  });
  it('penalty stage minimizers violate the constraint by λ⋆/μ; the augmented Lagrangian keeps μ = 10', () => {
    const pr = PRESETS.find((x) => x.id === 'outside-in')!;
    const p = P(pr.problem!);
    const lamStar = p.extra.multipliers[0][0];
    const [pen, al] = pr.methods!.map((m) => run(m.id, pr.problem!, m.params));
    const ends = (r: Result) =>
      r.trace.filter(
        (s, i) => i > 0 && (i + 1 === r.trace.length || r.trace[i + 1].info.outer !== s.info.outer),
      );
    const penEnds = ends(pen);
    for (const s of penEnds) {
      const mu = s.info.mu as number;
      // c(x(μ)) = λ(μ)/μ with λ(μ) → λ⋆ (N&W eq. 17.9): within 2 % once μ ≥ 100.
      if (mu >= 100)
        expect(Math.abs((s.info.violation as number) * mu - lamStar) / lamStar).toBeLessThan(0.02);
    }
    const last = penEnds[penEnds.length - 1];
    expect([last.info.mu, Number((last.info.violation as number).toExponential(1))]).toEqual([
      1e7, 1.2e-7,
    ]);
    expect(al.trace.every((s) => s.info.mu === 10)).toBe(true);
    expect(al.converged).toBe(true);
  });
  it('the barrier preset converges and SQP on the circle leaves the view', () => {
    const cp = PRESETS.find((x) => x.id === 'central-path')!;
    const b = run(cp.methods![0].id, cp.problem!, cp.methods![0].params);
    expect(b.converged).toBe(true);
    const kc = PRESETS.find((x) => x.id === 'kkt-circle')!;
    const p = P(kc.problem!);
    const s = run('sqp', p.id);
    expect(s.converged).toBe(true);
    const [[x0, x1], [y0, y1]] = p.domain;
    expect(
      s.trace.some(({ x }) => {
        const [a, c] = x as number[];
        return a < x0 || a > x1 || c < y0 || c > y1;
      }),
    ).toBe(true);
  });
});

describe('TeX the lab builds', () => {
  it('texNum', () => {
    expect(texNum(1000)).toBe('10^{3}');
    expect(texNum(0.5)).toBe('0.5');
    expect(texNum(1.23456e-7)).toBe('1.235 \\times 10^{-7}');
    expect(texNum(-2.5)).toBe('-2.5');
    expect(texNum(10)).toBe('10');
  });
  it('rules with the numbers of every step parse', () => {
    const cases: [string, string][] = [
      ['projected_gradient', 'hs21'],
      ['frank_wolfe', 'box_quadratic'],
      ['quadratic_penalty', 'quadratic_disk'],
      ['augmented_lagrangian', 'circle_eq'],
      ['log_barrier', 'hs21'],
      ['sqp', 'mishra_bird_constrained'],
    ];
    for (const [m, pid] of cases) {
      const rule = getMethod(m).doc!.rule;
      parses(rule);
      const r = run(m, pid);
      expect(ruleInstance(m, r.trace[0])).toBeNull();
      for (const s of r.trace.slice(0, 8)) parses(filledRule(m, rule, s));
    }
  });
  it('the rail formula of every problem splits into lines and parses', () => {
    for (const p of listProblems<ConstrainedProblem>('constrained')) {
      const tex = twoLineLatex(p.latex);
      expect(tex).toContain('\\text{s.t.}');
      parses(tex);
    }
  });
  it('MethodCard quantity labels parse', () => {
    for (const id of [
      'projected_gradient',
      'frank_wolfe',
      'quadratic_penalty',
      'augmented_lagrangian',
      'log_barrier',
      'sqp',
    ])
      for (const q of getMethod(id).doc!.quantities ?? []) parses(q.tex);
  });
});

describe('review fixes', () => {
  it('an active inequality with round-off c = 7e−15 at a converged SQP point is "active"', () => {
    const r = run('sqp', 'mishra_bird_constrained', { x0: [-8.5, -7.5] });
    expect(r.converged).toBe(true);
    const c = (r.trace.at(-1)!.info.constraints as number[])[0];
    expect(Math.abs(c)).toBeLessThan(1e-6);
    expect(constraintStatus('ineq', c)).toBe('active');
    expect(constraintStatus('ineq', 2e-6)).toBe('violated');
    expect(constraintStatus('ineq', -0.5)).toBe('inactive');
    expect(constraintStatus('eq', 3e-7)).toBe('satisfied');
  });
  it('λ⋆ and f⋆ come from the minimizer the run reaches, not always the global one', () => {
    const p = P('mishra_bird_constrained');
    const r = run('sqp', 'mishra_bird_constrained', { x0: [-8.5, -7.5] });
    const i = nearestMinimizer(p, r.x as number[]);
    expect(i).not.toBe(0);
    expect(Math.abs((r.fun as number) - p.extra.minima_f[i])).toBeLessThan(1e-6);
    const lam = (r.trace.at(-1)!.info.multipliers as number[])[0];
    expect(lam).toBeCloseTo(p.extra.multipliers[i][0], 3);
  });
  it('compact table numbers fit 7 characters (8 with a sign) and keep the exponent', () => {
    expect(compactNum(1.3e-5)).toBe('1.3e−5');
    expect(compactNum(1e-12)).toBe('1e−12');
    expect(compactNum(1e7)).toBe('1e7');
    expect(compactNum(-2.04e-9)).toBe('−2e−9');
    expect(compactNum(4.4e-16)).toBe('4e−16');
    expect(compactNum(-9.87e-11)).toBe('−1e−10');
    expect(compactNum(0.0031949)).toBe('0.0032');
    for (const v of [1.23e-15, -9.87e-11, 4.4e12, 0.5, 1024])
      expect(compactNum(v).length).toBeLessThanOrEqual(v < 0 ? 8 : 7);
  });
  it('clipSegment cuts a huge SQP step at the frame and keeps its direction', () => {
    const box: [number, number][] = [
      [-1, 1],
      [-1, 1],
    ];
    const c = clipSegment([0, 0], [-494, -833], box)!;
    expect(c.clipped).toBe(true);
    expect(c.from).toEqual([0, 0]);
    expect(c.to[1]).toBeCloseTo(-1, 12);
    expect(c.to[0] / c.to[1]).toBeCloseTo(494 / 833, 12);
    expect(clipSegment([0, 0], [0.5, 0.5], box)!.clipped).toBe(false);
  });
  it('the default SQP step 3 arrow ends inside the view and is labelled as clipped', () => {
    const p = P(DEFAULT_PROBLEM);
    const r = run('sqp', DEFAULT_PROBLEM);
    const g = stepGeometry('sqp', r.trace, 3, 0, p, null);
    const arrow = g.find((o) => o.kind === 'arrow');
    expect(arrow && arrow.kind === 'arrow').toBe(true);
    if (arrow?.kind !== 'arrow') return;
    const [[x0, x1], [y0, y1]] = p.domain;
    expect(arrow.to[0]).toBeGreaterThanOrEqual(x0);
    expect(arrow.to[0]).toBeLessThanOrEqual(x1);
    expect(arrow.to[1]).toBeGreaterThanOrEqual(y0);
    expect(arrow.to[1]).toBeLessThanOrEqual(y1);
    expect(JSON.stringify(arrow.label)).toContain('clipped');
    // The linearized constraints are not overlays in the method color (the lab draws them as
    // dotted ink lines); only their labels are.
    expect(g.some((o) => o.kind === 'segment')).toBe(false);
    const lin = linearizedConstraints(r.trace, 3, p);
    expect(lin).toHaveLength(1);
  });
  it('the x⋆ label goes to the side away from the KKT arrows', () => {
    const dom: [number, number][] = [
      [-1.5, 1.5],
      [-1.5, 1.5],
    ];
    const arrow = (to: [number, number]) => [{ from: [0, 0] as [number, number], to, weight: 1.5 }];
    expect(minimizerLabelSide([0, 0], dom, arrow([1, 0]))).not.toBe('right');
    expect(minimizerLabelSide([0, 0], dom, arrow([-1, 0]))).toBe('right');
    expect(minimizerLabelSide([0, 0], dom, arrow([0, 1]))).toBe('right');
    // A path coming in from the left along the axis and an arrow to the right: above or below.
    const both = [
      ...arrow([1, 0]),
      { from: [-1.2, 0] as [number, number], to: [0, 0] as [number, number], weight: 1 },
    ];
    expect(['above', 'below']).toContain(minimizerLabelSide([0, 0], dom, both));
    // Near the right edge the label never goes right.
    expect(minimizerLabelSide([1.4, 0], dom, arrow([-1, 0]))).not.toBe('right');
  });
  it('hs21 opens on a view that holds x⋆, the default start and every default path', () => {
    const p = P('hs21');
    const [[x0, x1], [y0, y1]] = VIEW_DOMAIN.hs21;
    const inside = (q: readonly number[]) => q[0] >= x0 && q[0] <= x1 && q[1] >= y0 && q[1] <= y1;
    expect(inside(p.minima[0])).toBe(true);
    expect(inside(p.x0 as number[])).toBe(true);
    for (const id of ['projected_gradient', 'log_barrier', 'sqp'])
      expect(run(id, 'hs21').trace.every((s) => inside(s.x as number[]))).toBe(true);
  });
});
