/**
 * The least-squares lab's drawn geometry is the method's mathematics:
 *   - 𝐡(μ) recomputed at 𝐱ₖ with μₖ is the LM trial step in Step.info (so the arrow tip lies on
 *     the dashed curve 𝐡(μ) and on the trust disk), and 𝐡(0) is the GN step 𝐩ₖ;
 *   - the LM step minimizes the GN model on its disk: the model ellipse through 𝐱ₖ + 𝐡ₖ is
 *     tangent there (the model gradient is parallel to 𝐡ₖ, pointing inward);
 *   - the GN ellipse passes through 𝐱ₖ; circle feet lie on the circle;
 *   - the substituted rules and TeX numbers.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { reviveNumbers } from '../../src/core/json';
import { runMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { LeastSquaresProblem } from '../../src/problems/least_squares';
import '../../src/problems/least_squares';
import '../../src/methods/unconstrained/least_squares';
import {
  circleFoot,
  dataSpaceFor,
  hOfMu,
  linearization,
  paramNames,
  stepGeometry,
} from '../../src/labs/least-squares/models';
import { roundingNoise, stepRuleTex, texNum, texVec } from '../../src/labs/least-squares/text';
import { compactNum, columnsFor } from '../../src/labs/least-squares/columns';
import { exitPoint, sideWords } from '../../src/labs/least-squares/offview';
import { curveRange } from '../../src/labs/least-squares/models';
import {
  DEFAULT_PROBLEM,
  DEFAULT_SELECTION,
  LAB_START,
  PRESETS,
} from '../../src/labs/least-squares/presets';

const P = (id: string) => getProblem<LeastSquaresProblem>(id);
const close = (a: readonly number[], b: readonly number[], tol: number) =>
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i]))));

describe('step geometry', () => {
  for (const id of ['exp_decay_fit', 'rosenbrock_ls', 'circle_fit', 'michaelis_menten']) {
    it(`LM trial steps lie on 𝐡(μ) and the model is tangent to the disk (${id})`, () => {
      const p = P(id);
      const res = runMethod('levenberg_marquardt', p, {});
      res.trace.slice(0, -1).forEach((s, k) => {
        const next = res.trace[k + 1];
        const h = next.info.step as number[];
        const mu = next.info.lambda as number;
        const lin = linearization(p, s.x as number[])!;
        close(hOfMu(lin, mu), h, 1e-8);
        // ∇L(𝐡) = Jᵀ(𝐫 + J𝐡) = −μ𝐡 at the LM step (KKT of the trust-region subproblem).
        const Jh = lin.J.map((row) => row[0] * h[0] + row[1] * h[1]);
        const gradL = [0, 1].map((a) =>
          lin.J.reduce((acc, row, i) => acc + row[a] * (lin.r[i] + Jh[i]), 0),
        );
        close(gradL, [-mu * h[0], -mu * h[1]], 1e-6);
        const g = stepGeometry(p, res.trace, k, 1, 'levenberg_marquardt');
        expect(g.radius).toBeCloseTo(Math.hypot(h[0], h[1]), 12);
        expect(g.overlays.some((o) => o.kind === 'disk')).toBe(true);
      });
    });
    it(`the GN point is 𝐡(0) and the GN ellipse passes through 𝐱ₖ (${id})`, () => {
      const p = P(id);
      const res = runMethod('gauss_newton', p, {});
      res.trace.slice(0, -1).forEach((s, k) => {
        const pk = res.trace[k + 1].info.step as number[];
        const lin = linearization(p, s.x as number[])!;
        close(lin.p!, pk, 1e-8);
        const g = stepGeometry(p, res.trace, k, 0, 'gauss_newton');
        const ell = g.overlays.find((o) => o.kind === 'ellipse');
        if (ell && ell.kind === 'ellipse') {
          const x = s.x as number[];
          const d = [x[0] - ell.center[0], x[1] - ell.center[1]];
          const M = ell.matrix;
          const q = M[0][0] * d[0] ** 2 + 2 * M[0][1] * d[0] * d[1] + M[1][1] * d[1] ** 2;
          expect(Math.abs(Math.sqrt(q) - (ell.radius ?? 1))).toBeLessThan(
            1e-8 * (1 + (ell.radius ?? 1)),
          );
        }
      });
    });
  }

  it('no geometry at the last iterate', () => {
    const p = P('exp_decay_fit');
    const res = runMethod('gauss_newton', p, {});
    expect(stepGeometry(p, res.trace, res.trace.length - 1, 0, 'gauss_newton').overlays).toEqual(
      [],
    );
  });

  it('rejected LM trials are dashed and keep 𝐱', () => {
    const p = P('exp_decay_fit');
    const res = runMethod('levenberg_marquardt', p, { x0: [0.2, 2.8] });
    const k = res.trace.findIndex((_, i) => res.trace[i + 1]?.info.accepted === false);
    expect(k).toBeGreaterThanOrEqual(0);
    const g = stepGeometry(p, res.trace, k, 1, 'levenberg_marquardt');
    expect(g.accepted).toBe(false);
    const arrow = g.overlays.find((o) => o.kind === 'arrow');
    expect(arrow && arrow.kind === 'arrow' && arrow.dashed).toBe(true);
    expect(res.trace[k + 1].x).toEqual(res.trace[k].x);
  });
});

describe('data space', () => {
  it('kinds, parameter names and models', () => {
    expect(dataSpaceFor(P('exp_decay_fit')).kind).toBe('curve');
    expect(dataSpaceFor(P('michaelis_menten')).kind).toBe('curve');
    expect(dataSpaceFor(P('circle_fit')).kind).toBe('circle');
    expect(dataSpaceFor(P('rosenbrock_ls')).kind).toBe('residual');
    expect(paramNames(P('michaelis_menten'))).toEqual(['V', 'K']);
    // r = model − data, so the curve's residuals are the problem's residuals.
    for (const id of ['exp_decay_fit', 'michaelis_menten']) {
      const p = P(id);
      const s = dataSpaceFor(p);
      if (s.kind !== 'curve') throw new Error('curve expected');
      const x = p.x0;
      close(
        s.t.map((t, i) => s.model(t, x) - s.y[i]),
        p.residual(x),
        1e-12,
      );
    }
  });
  it('circle feet lie on the circle, on the ray through the point', () => {
    const c: [number, number] = [1, -0.5];
    const f = circleFoot(c, 2, [4, 3.5]);
    expect(Math.hypot(f[0] - c[0], f[1] - c[1])).toBeCloseTo(2, 12);
    expect((f[0] - c[0]) * (3.5 - c[1]) - (f[1] - c[1]) * (4 - c[0])).toBeCloseTo(0, 12);
  });
});

describe('TeX', () => {
  it('numbers and vectors', () => {
    expect(texNum(0.004146)).toBe('0.004146');
    expect(texNum(7.12e-4, 3)).toBe('7.12\\times10^{-4}');
    expect(texNum(1e-8)).toBe('10^{-8}');
    expect(texNum(Infinity)).toBe('\\infty');
    expect(texNum(null)).toBe('\\text{—}');
    expect(texVec([2.5, -1])).toBe('(2.5,\\,-1)');
  });
  it('substituted rules name the step leaving 𝐱ₖ', () => {
    const p = P('exp_decay_fit');
    const gn = runMethod('gauss_newton', p, { x0: [0.2, 2.8] });
    const t0 = stepRuleTex('gauss_newton', gn.trace, 0)!;
    expect(t0).toContain('\\mathbf{p}_{0}');
    expect(t0).toContain('\\mathbf{x}_{1}');
    const lm = runMethod('levenberg_marquardt', p, { x0: [0.2, 2.8] });
    expect(stepRuleTex('levenberg_marquardt', lm.trace, 0)).toContain('\\text{reject: }');
    const last = stepRuleTex('levenberg_marquardt', lm.trace, lm.trace.length - 1)!;
    expect(last).toContain('\\nabla f');
    expect(stepRuleTex('gauss_newton', gn.trace, 99)).toBeNull();
  });
});

// ── Presets against Python ─────────────────────────────────────────────────────────────

type Raw = Record<string, unknown>;
interface PresetCase {
  preset: string;
  method: string;
  problem: string;
  params: Raw;
  result: Raw;
}
const FIX = reviveNumbers(
  JSON.parse(
    readFileSync(
      fileURLToPath(new URL('./fixtures/least_squares_python.json', import.meta.url)),
      'utf8',
    ),
  ),
) as { presets: PresetCase[] };

/** The lab's view as (preset id, problem, start, selection). */
const VIEWS = [
  {
    id: 'default',
    problem: DEFAULT_PROBLEM,
    start: LAB_START[DEFAULT_PROBLEM],
    methods: DEFAULT_SELECTION,
  },
  ...PRESETS.map((p) => ({
    id: p.id,
    problem: p.problem!,
    start: p.start as [number, number],
    methods: p.methods!,
  })),
];

describe('the first view and the presets against Python', () => {
  it('every method of every view is a fixture case with the same inputs', () => {
    for (const v of VIEWS)
      for (const m of v.methods) {
        const c = FIX.presets.find((q) => q.preset === v.id && q.method === m.id);
        expect(c, `${v.id} / ${m.id}`).toBeDefined();
        expect(c!.problem).toBe(v.problem);
        expect(c!.params).toEqual({ x0: v.start, ...m.params });
      }
  });

  for (const c of FIX.presets) {
    it(`${c.preset}: ${c.method} matches Python`, () => {
      const got = runMethod(c.method, P(c.problem), c.params);
      const want = c.result;
      const wx = want.x as number[];
      const wtrace = want.trace as Raw[];
      expect(got.converged).toBe(want.converged);
      // The traces may part only at the noise floor: once f has stopped changing beyond 10⁻¹²
      // relative, the gain ratio ϱ is a ratio of rounding errors (mirror/LM parts at k = 9,
      // where NumPy gets ϱ = 0 and rejects, the port gets ϱ = 31.5 and accepts).
      const fEnd = want.fun as number;
      let split = -1;
      got.trace.forEach((s, k) => {
        const w = wtrace[k];
        if (split >= 0 || !w) return;
        if (Math.abs(s.fun! - (w.fun as number)) > 1e-9 * (1 + Math.abs(w.fun as number)))
          split = k;
        else if (s.info.accepted !== (w.info as Raw).accepted) split = k;
      });
      // The rank preset's last f is rounding noise itself (checked below).
      const rankNoise = c.preset === 'rank' && c.method === 'gauss_newton';
      if (rankNoise) expect(split).toBe(got.trace.length - 1);
      if (split >= 0 && !rankNoise) {
        const fPrev = wtrace[split - 1].fun as number;
        expect(Math.abs(fPrev - fEnd), `parts at k = ${split}`).toBeLessThanOrEqual(
          1e-12 * Math.abs(fEnd),
        );
      } else {
        expect(got.nIter).toBe(want.n_iter);
        expect(got.nFev).toBe(want.n_fev);
        // Iterates before the last: within 1e-8 relative.
        got.trace
          .slice(0, -1)
          .forEach((s, k) => close(s.x as number[], wtrace[k].x as number[], 1e-8));
      }
      const prev = got.trace[got.trace.length - 2]?.x as number[] | undefined;
      const noise = roundingNoise(got.x as number[], prev);
      if (rankNoise) {
        // The documented noise: the second undamped step cancels a to rounding level. Both land
        // within 64ε·|a₁| of 0, with different signs (−1.6e−14 here, +1.8e−15 in NumPy); b
        // agrees, and f(𝐱₂) — dominated by a·e^{9.04 t} — differs by orders of magnitude.
        expect(noise).toEqual([0]);
        expect(roundingNoise(wx, wtrace[1].x as number[])).toEqual([0]);
        close([(got.x as number[])[1]], [wx[1]], 1e-10);
        expect(Math.sign((got.x as number[])[0])).not.toBe(Math.sign(wx[0]));
        expect(got.fun! / (want.fun as number)).toBeGreaterThan(2);
        // The step rule says so, and prints neither the sign nor f.
        const tex = stepRuleTex('gauss_newton', got.trace, got.trace.length - 1, ['a', 'b'])!;
        expect(tex).toMatch(/\\mathcal\{O\}\(10\^\{-1[3-6]\}\)/);
        expect(tex).toContain('the sign is noise');
        expect(tex).not.toContain(texNum(got.fun));
      } else {
        expect(noise).toEqual([]);
        close(got.x as number[], wx, 1e-6);
        expect(Math.abs(got.fun! - (want.fun as number))).toBeLessThanOrEqual(
          1e-9 * (1 + Math.abs(want.fun as number)),
        );
      }
    });
  }

  const ref = (preset: string, method: string) =>
    FIX.presets.find((q) => q.preset === preset && q.method === method)!.result;
  it('the numbers the notes state', () => {
    // newton: GN leaves the view at (1, −3.84) and lands on (1, 1) in two steps.
    const nw = ref('newton', 'gauss_newton');
    const x1 = (nw.trace as Raw[])[1].x as number[];
    expect(x1[0]).toBeCloseTo(1, 2);
    expect(x1[1]).toBeCloseTo(-3.84, 2);
    expect(nw.n_iter).toBe(2);
    close(nw.x as number[], [1, 1], 1e-12);
    // rank: the first undamped step overshoots to (2.45, −9.04) with f ≈ 3·10³³.
    const rk = (ref('rank', 'gauss_newton').trace as Raw[])[1];
    close(rk.x as number[], [2.45, -9.04], 1e-3);
    expect(rk.fun as number).toBeGreaterThan(1e33);
    expect(ref('rank', 'levenberg_marquardt').converged).toBe(true);
    // mirror: both end at (1.30, 2.74), f = 1.19, about 200 × the global minimum.
    const fStar = P('circle_fit').extra.f_min;
    for (const m of ['gauss_newton', 'levenberg_marquardt']) {
      const r = ref('mirror', m);
      close(r.x as number[], [1.3, 2.74], 5e-3);
      expect(r.fun as number).toBeCloseTo(1.19, 2);
      expect((r.fun as number) / fStar).toBeGreaterThan(150);
      expect((r.fun as number) / fStar).toBeLessThan(250);
    }
    // scale: 27 LM iterations against 7 for GN.
    expect(ref('scale', 'levenberg_marquardt').n_iter).toBe(27);
    expect(ref('scale', 'gauss_newton').n_iter).toBe(7);
  });
});

describe('table, data space and off-view helpers', () => {
  it('compact numbers fit a narrow cell', () => {
    expect(compactNum(4.65e-4)).toBe('4.7e−4');
    expect(compactNum(0.98894, 3)).toBe('0.989');
    expect(compactNum(-7.518e32, 3)).toBe('−7.5e32');
    expect(compactNum(-1534, 3)).toBe('−1.5e3');
    expect(compactNum(1e-8)).toBe('1e−8');
    expect(compactNum(12, 2)).toBe('12');
    expect(compactNum(null)).toBe('—');
    for (const v of [1.234e-17, -9.99e99, 0.0123, 123])
      expect(compactNum(v, 3).length).toBeLessThanOrEqual(7);
  });
  it('the LM table puts the accept/reject mark in the gain cell, inside a 380 px panel', () => {
    const cols = columnsFor('levenberg_marquardt');
    expect(cols.map((c) => c.key)).toEqual(['k', 'x', 'fun', 'mu', 'gain']);
    const min = cols.reduce((s, c) => s + Number(/(\d+)px/.exec(c.width ?? '80px')![1]), 0);
    expect(min + 10 * (cols.length - 1) + 24).toBeLessThanOrEqual(380);
    const p = P('exp_decay_fit');
    const lm = runMethod('levenberg_marquardt', p, { x0: [0.2, 2.8] });
    const rows = lm.trace.map((s, k, tr) => ({
      ...s,
      info: { ...s.info, next: tr[k + 1]?.info ?? null },
    }));
    const gain = cols[4];
    expect(String(gain.value(rows[0]))).toMatch(/ ×$/);
    expect(rows.some((r) => / ✓$/.test(String(gain.value(r))))).toBe(true);
    expect(gain.value(rows[rows.length - 1])).toBe('—');
  });
  it('the model curve starts at t = max(0, t_min), never inside the padded strip', () => {
    expect(curveRange(0, [-0.1, 4.3])).toEqual([0, 4.3]);
    expect(curveRange(0.5, [-0.2, 4.3])).toEqual([0.5, 4.3]);
  });
  it('exit points and side words', () => {
    const r = { left: 0, top: 0, right: 100, bottom: 100 };
    const e = exitPoint(
      [
        [50, 50],
        [50, 250],
      ],
      r,
    )!;
    expect(e.p).toEqual([50, 100]);
    expect(e.dir).toEqual([0, 1]);
    expect(
      exitPoint(
        [
          [150, 50],
          [50, 50],
        ],
        r,
      ),
    ).toBeNull();
    expect(
      exitPoint(
        [
          [10, 10],
          [20, 20],
        ],
        r,
      ),
    ).toBeNull();
    expect(sideWords([50, 250], r)).toBe('below the view');
    expect(sideWords([-5, -5], r)).toBe('above and left of the view');
  });
  it('the default first frame: the GN point and three failed trials lie below the domain', () => {
    const p = P('exp_decay_fit');
    const gn = runMethod('gauss_newton', p, { x0: [0.2, 2.8] });
    const g = stepGeometry(p, gn.trace, 0, 0, 'gauss_newton');
    const mark = g.marks.find((m) => m.role === 'gn')!;
    expect(mark.at[1]).toBeLessThan(p.domain[1][0]);
    // α = ½, ¼, ⅛ failed (α = 1 is the GN point itself); ½ and ¼ lie far below b = 0.
    expect(mark.trials).toHaveLength(3);
    expect(mark.trials!.filter((t) => t[1] < p.domain[1][0]).length).toBe(3);
  });
});
