/**
 * Every claim in the lab's defaults and "Try this" presets, checked against the TS ports.
 */
import { describe, expect, it } from 'vitest';
import './setup';
import { defaults, getMethod } from '../../core/registry';
import { getProblem } from '../../problems/registry';
import type { Problem2D, Result } from '../../core/types';
import { DEFAULT_PROBLEM, DEFAULT_SELECTION, PRESETS, ZIGZAG_X0 } from './presets';

function runPreset(id: string) {
  const p = PRESETS.find((x) => x.id === id)!;
  const problem = getProblem<Problem2D>(p.problem!);
  const out: Record<string, Result> = {};
  for (const m of p.methods!) {
    const method = getMethod<Problem2D>(m.id);
    const x0 = (p.start as number[] | undefined) ?? (problem.x0 as number[]);
    out[m.id] = method.fn(problem, { ...defaults(method.spec), ...m.params, x0 }) as Result;
  }
  return { problem, out };
}

const last = (r: Result) => r.trace[r.trace.length - 1].x as number[];

describe('default view', () => {
  it('BFGS 38, dogleg 24 and Nelder–Mead 110 iterations on Rosenbrock, all converged', () => {
    const problem = getProblem<Problem2D>(DEFAULT_PROBLEM);
    const n = DEFAULT_SELECTION.map((s) => {
      const m = getMethod<Problem2D>(s.id);
      const r = m.fn(problem, { ...defaults(m.spec), x0: problem.x0 as number[] }) as Result;
      expect(r.converged).toBe(true);
      return r.nIter;
    });
    expect(n).toEqual([38, 24, 110]);
  });
});

describe('presets', () => {
  it('ids are unique and every preset names its focus', () => {
    expect(new Set(PRESETS.map((p) => p.id)).size).toBe(PRESETS.length);
    for (const p of PRESETS) expect(p.methods!.map((m) => m.id)).toContain(p.extra!.f);
  });

  it('maximizer: pure Newton stops at the maximum f = 181.6; damped Newton and TR reach (3, 2)', () => {
    const { out } = runPreset('maximizer');
    const pn = out.pure_newton;
    expect(pn.converged).toBe(false);
    expect(pn.message).toMatch(/maximizer/);
    expect(pn.fun!).toBeCloseTo(181.6165, 3);
    for (const id of ['damped_newton', 'trust_region_exact']) {
      expect(out[id].converged).toBe(true);
      expect(last(out[id])[0]).toBeCloseTo(3, 6);
      expect(last(out[id])[1]).toBeCloseTo(2, 6);
    }
  });

  it('zigzag: 382 orthogonal exact steps with ratio ((κ−1)/(κ+1))²; CG ends in 2', () => {
    const { out } = runPreset('zigzag');
    const gd = out.gradient_descent;
    expect(gd.converged).toBe(true);
    expect(gd.nIter).toBe(382);
    const t = gd.trace;
    for (let k = 1; k < 6; k++) {
      const a = (t[k].x as number[]).map((v, i) => v - (t[k - 1].x as number[])[i]);
      const b = (t[k + 1].x as number[]).map((v, i) => v - (t[k].x as number[])[i]);
      expect(Math.abs(a[0] * b[0] + a[1] * b[1])).toBeLessThan(1e-12);
      expect(t[k + 1].fun! / t[k].fun!).toBeCloseTo((49 / 51) ** 2, 9);
    }
    expect((49 / 51) ** 2).toBeCloseTo(0.923, 3);
    expect(out.cg_fletcher_reeves.converged).toBe(true);
    expect(out.cg_fletcher_reeves.nIter).toBe(2);
    expect(ZIGZAG_X0).toEqual([2.364, 1.848]);
  });

  it('offview: Newton x₂ = (0.76, −3.18); the dogleg rejects that step (ρ = −0.41), Δ → ¼, 24 steps', () => {
    const { out, problem } = runPreset('offview');
    const x2 = out.pure_newton.trace[2].x as number[];
    expect(x2[0]).toBeCloseTo(0.763, 3);
    expect(x2[1]).toBeCloseTo(-3.175, 3);
    expect(x2[1]).toBeLessThan(problem.domain[1][0]);
    const tr = out.trust_region_dogleg;
    const s2 = tr.trace[2].info;
    expect(s2.accepted).toBe(false);
    expect(s2.rho as number).toBeCloseTo(-0.41, 2);
    expect(s2.new_radius).toBe(0.25);
    const np = s2.newton_point as number[];
    const nx = out.pure_newton.trace[2].x as number[];
    expect(Math.hypot(np[0] - nx[0], np[1] - nx[1])).toBeLessThan(1e-9);
    expect(tr.converged).toBe(true);
    expect(tr.nIter).toBe(24);
  });

  it('adamw: Adam converges in 281; AdamW keeps ‖∇f‖ ≈ 1.7×10⁻³ through 5,000 steps', () => {
    const { out, problem } = runPreset('adamw');
    expect(out.adam.converged).toBe(true);
    expect(out.adam.nIter).toBe(281);
    expect(out.adamw.converged).toBe(false);
    expect(out.adamw.nIter).toBe(5000);
    const g = problem.grad(last(out.adamw));
    expect(Math.hypot(...g)).toBeCloseTo(1.7e-3, 4);
  });

  it('schedules: GD with 1/L 639 steps, silver (κ-aware) 239 with steps up to 22.9/L, FISTA 92', () => {
    const { out } = runPreset('schedules');
    expect(out.gradient_descent.converged).toBe(true);
    expect(out.gradient_descent.nIter).toBe(639);
    const s = out.silver_gd_strongly_convex;
    expect(s.converged).toBe(true);
    expect(s.nIter).toBe(239);
    expect(s.extra.L).toBe(50);
    // lr = 0.02 is exactly 1/L.
    expect(1 / (s.extra.L as number)).toBe(0.02);
    const hMax = Math.max(...s.trace.slice(1).map((t) => t.info.h as number));
    expect(hMax).toBeCloseTo(22.9, 1);
    expect(out.fista.converged).toBe(true);
    expect(out.fista.nIter).toBe(92);
  });

  it('aa-saddle: AA stops at the saddle (0.087, 2.884) after 16; GD 21, ARC 10 reach (3, 2)', () => {
    const { out } = runPreset('aa-saddle');
    const aa = out.anderson_gd;
    expect(aa.converged).toBe(false);
    expect(aa.nIter).toBe(16);
    expect(aa.message).toMatch(/saddle point/);
    expect(last(aa)[0]).toBeCloseTo(0.087, 3);
    expect(last(aa)[1]).toBeCloseTo(2.884, 3);
    expect(out.gradient_descent.nIter).toBe(21);
    expect(out.arc.nIter).toBe(10);
    for (const id of ['gradient_descent', 'arc']) {
      expect(out[id].converged).toBe(true);
      expect(last(out[id])[0]).toBeCloseTo(3, 6);
      expect(last(out[id])[1]).toBeCloseTo(2, 6);
    }
  });

  it('arc-escape: Newton (2) and super-universal (3) stop at the saddle; ARC reaches (−0.090, 0.713) in 10', () => {
    const { out } = runPreset('arc-escape');
    expect(out.pure_newton.converged).toBe(false);
    expect(out.pure_newton.nIter).toBe(2);
    expect(out.pure_newton.message).toMatch(/saddle point/);
    expect(out.reg_newton.converged).toBe(false);
    expect(out.reg_newton.nIter).toBe(3);
    expect(out.reg_newton.message).toMatch(/saddle point/);
    for (const id of ['pure_newton', 'reg_newton'])
      expect(Math.hypot(...last(out[id]))).toBeLessThan(1e-6);
    expect(out.arc.converged).toBe(true);
    expect(out.arc.nIter).toBe(10);
    expect(last(out.arc)[0]).toBeCloseTo(-0.0898, 4);
    expect(last(out.arc)[1]).toBeCloseTo(0.7127, 4);
  });
});
