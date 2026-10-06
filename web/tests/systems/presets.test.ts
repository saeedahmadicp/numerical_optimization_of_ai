/**
 * The "Try this" notes of the systems lab state facts about runs; this test runs each preset and
 * checks every number its note gives (src/labs/systems/presets.ts).
 */
import { describe, expect, it } from 'vitest';
import { runMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Result, Vector } from '../../src/core/types';
import type { SystemProblem } from '../../src/problems/systems';
import '../../src/methods/roots/systems';
import '../../src/problems/systems';
import { PRESETS } from '../../src/labs/systems/presets';

const preset = (id: string) => PRESETS.find((p) => p.id === id)!;

function run(id: string, slot = 0, extra: Record<string, unknown> = {}): Result {
  const p = preset(id);
  const sel = p.methods![slot];
  const prob = getProblem<SystemProblem>(p.problem!);
  const x0 = (p.start as number[] | undefined) ?? (prob.x0 as number[]);
  return runMethod(sel.id, prob, { ...sel.params, ...extra, x0 });
}

const xs = (r: Result) => r.trace.map((s) => s.x as Vector);
const close = (a: number, b: number, tol: number) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);

describe('systems presets: the notes state facts of the runs', () => {
  it('every preset that does not choose the focus clears it', () => {
    for (const p of PRESETS) expect(p.extra && 'fm' in p.extra).toBe(true);
  });

  it('leaves: nine steps mostly below the window, a jump from x + y ≈ 0.019 to a far root', () => {
    const r = run('leaves');
    const prob = getProblem<SystemProblem>('trig_system');
    const yMin = prob.domain[1][0];
    const path = xs(r);
    const below = path.slice(1, 10).filter((x) => x[1] < yMin).length;
    expect(below).toBe(8); // only 𝐱₆ = (0.41, −0.70) is back inside
    const s9 = path[9][0] + path[9][1];
    close(s9, 0.019, 5e-4);
    close(Math.sin(s9), 0.02, 2e-3);
    expect(r.converged).toBe(true);
    const x = r.x as Vector;
    close(x[0], (5 * Math.PI) / 6 - 6 * Math.PI, 1e-9);
    close(x[1], (5 * Math.PI) / 6 + 4 * Math.PI, 1e-9);
    // With damping: (π/6, π/6) in 7 steps.
    const d = run('leaves', 0, { damping: true });
    expect(d.converged).toBe(true);
    expect(d.nIter).toBe(7);
    close((d.x as Vector)[0], Math.PI / 6, 1e-9);
    close((d.x as Vector)[1], Math.PI / 6, 1e-9);
  });

  it('trap: the line search fails at (13.55, −0.897) on the singular line, ½‖F‖² = 29.1', () => {
    const r = run('trap');
    expect(r.converged).toBe(false);
    expect(r.message).toMatch(/^line search failed/);
    const [x, y] = r.x as Vector;
    close(x, 13.55, 5e-3);
    close(y, -0.897, 5e-4);
    close(6 * y * y - 8 * y - 12, 0, 5e-3); // det J
    close(0.5 * r.fun! * r.fun!, 29.1, 0.05);
    // The non-root minimum of ½‖F‖²: for fixed y the best x makes F₁ = −F₂, so
    // φ(y) = (F₁ − F₂)²/4 with F₁ − F₂ = 16 − 2y³ + 4y² + 12y (independent of x).
    const prob = getProblem<SystemProblem>('freudenstein_roth');
    let best = { y: 0, phi: Infinity };
    for (let yy = -1.2; yy <= -0.6; yy += 1e-6) {
      const d = 16 - 2 * yy ** 3 + 4 * yy ** 2 + 12 * yy;
      if (d * d < best.phi) best = { y: yy, phi: d * d };
    }
    const F0 = prob.components[0](0, best.y);
    const F1 = prob.components[1](0, best.y);
    const xm = -(F0 + F1) / 2; // F₁ + F₂ = 0
    close(best.y, -0.897, 5e-4);
    close(xm, 11.41, 5e-3);
    close(best.phi / 4, 24.49, 0.01);
    expect(Math.hypot(...prob.f([xm, best.y]))).toBeCloseTo(Math.sqrt(best.phi / 2), 6);
    // Without damping: the root (5, 4) in 43 steps.
    const pure = run('trap', 0, { damping: false });
    expect(pure.converged).toBe(true);
    expect(pure.nIter).toBe(43);
    close((pure.x as Vector)[0], 5, 1e-9);
    close((pure.x as Vector)[1], 4, 1e-9);
  });

  it('drift: Newton in 7 steps; Broyden spends 100 and stops at (−0.31, 0.09), ‖F‖ = 3.5', () => {
    const n = run('drift', 0);
    expect(n.converged).toBe(true);
    expect(n.nIter).toBe(7);
    const b = run('drift', 1);
    expect(b.converged).toBe(false);
    expect(b.nIter).toBe(100);
    close((b.x as Vector)[0], -0.31, 5e-3);
    close((b.x as Vector)[1], 0.09, 5e-3);
    close(b.fun!, 3.5, 0.05);
  });
});
