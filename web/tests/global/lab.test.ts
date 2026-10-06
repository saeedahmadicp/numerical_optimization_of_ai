/**
 * Global lab: the claims of the default view and of every "Try this" preset hold for the runs
 * they set (seed included), and the pure geometry / table / equation helpers are right on every
 * step of every method.
 */
import { describe, expect, it } from 'vitest';
import { defaults, getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Problem, Result, Step, Vector } from '../../src/core/types';
import '../../src/labs/global/setup';
import { DEFAULT_SELECTION, PRESETS, SEED_KEY } from '../../src/labs/global/presets';
import {
  GLOBAL_METHODS,
  acceptedCount,
  clampToBox,
  deDonors,
  deHighlight,
  gaussianEllipseMatrix,
  globalMin,
  lerpCov,
  phaseAt,
  searchScale,
  temperatureOf,
  trackPoint,
} from '../../src/labs/global/geometry';
import {
  cellSpans,
  decimalsFor,
  meanSum,
  metropolisTex,
  sameMinimum,
  stepQuantities,
  stepTex,
  tn,
  tp,
} from '../../src/labs/global/stepTex';
import { rowVaries } from '../../src/labs/global/useStartWhenDrawn';
import { deRule } from '../../src/methods/unconstrained/global_';
import { iterationColumns, populationView } from '../../src/labs/global/tables';
import { trackAlpha, TRACK_FADE } from '../../src/labs/global/draw';

function run(id: string, problem: string, params: Record<string, unknown>, seed = 0): Result {
  const { spec, fn } = getMethod(id);
  return fn(getProblem(problem), { ...defaults(spec), ...params, seed } as never);
}

const presetRun = (presetId: string, methodId: string): Result => {
  const p = PRESETS.find((x) => x.id === presetId)!;
  const m = p.methods!.find((x) => x.id === methodId)!;
  return run(methodId, p.problem!, m.params, Number(p.extra![SEED_KEY]));
};

describe('default view and presets say what the runs do', () => {
  it('default: the swarm reaches the origin; annealing and CMA-ES settle in side basins', () => {
    const r = Object.fromEntries(
      DEFAULT_SELECTION.map((s) => [s.id, run(s.id, 'rastrigin', s.params)]),
    );
    expect(r.particle_swarm.fun).toBeLessThan(1e-10);
    expect(r.simulated_annealing.converged).toBe(true);
    expect(r.simulated_annealing.nIter).toBeGreaterThan(400);
    expect(r.simulated_annealing.nIter).toBeLessThan(500);
    expect(r.simulated_annealing.fun).toBeGreaterThan(0.9);
    expect(Math.abs((r.cma_es.fun as number) - 1)).toBeLessThan(0.05);
  });

  it('side-basins: CMA-ES (λ = 6) and DE/best/1 stop at f ≈ 1, the swarm at 0', () => {
    expect(Math.abs((presetRun('side-basins', 'cma_es').fun as number) - 1)).toBeLessThan(0.05);
    expect(
      Math.abs((presetRun('side-basins', 'differential_evolution').fun as number) - 1),
    ).toBeLessThan(0.05);
    expect(presetRun('side-basins', 'particle_swarm').fun).toBeLessThan(1e-10);
  });

  it('rosenbrock-ellipse: C becomes elongated along the valley, then the run reaches (1, 1)', () => {
    const r = presetRun('rosenbrock-ellipse', 'cma_es');
    expect(r.converged).toBe(true);
    expect(Math.max(...(r.x as Vector).map((v) => Math.abs(v - 1)))).toBeLessThan(1e-6);
    // Somewhere on the way, κ(C) ≥ 10 with the major axis along the valley tangent (1, 2x).
    const aligned = r.trace.some((s) => {
      const C = s.info.covariance as number[][];
      const m = s.info.mean as Vector;
      const tr = C[0][0] + C[1][1];
      const det = C[0][0] * C[1][1] - C[0][1] ** 2;
      const l1 = tr / 2 + Math.sqrt((tr * tr) / 4 - det);
      const l2 = tr / 2 - Math.sqrt((tr * tr) / 4 - det);
      const v = [C[0][1], l1 - C[0][0]];
      const t = [1, 2 * m[0]];
      const cos =
        Math.abs(v[0] * t[0] + v[1] * t[1]) / (Math.hypot(v[0], v[1]) * Math.hypot(t[0], t[1]));
      return l1 / l2 >= 10 && cos > 0.95;
    });
    expect(aligned).toBe(true);
  });

  it('fast-cooling: α = 0.9 freezes near (−1, −1) at f ≈ 2; basin hopping reaches 0', () => {
    const sa = presetRun('fast-cooling', 'simulated_annealing');
    expect(sa.converged).toBe(true);
    expect(Math.abs((sa.fun as number) - 2)).toBeLessThan(0.05);
    const x = sa.x as Vector;
    expect(Math.abs(x[0] + 1)).toBeLessThan(0.05);
    expect(Math.abs(x[1] + 1)).toBeLessThan(0.05);
    expect(Math.log(10) / Math.log(1 / 0.9)).toBeCloseTo(21.85, 1); // "tenfold every 22 steps"
    expect(presetRun('fast-cooling', 'basin_hopping').fun).toBeLessThan(1e-10);
  });

  it('log-cooling: still hot (T ≈ 0.91) at the 2,000-step budget', () => {
    const r = presetRun('log-cooling', 'simulated_annealing');
    expect(r.converged).toBe(false);
    expect(r.nIter).toBe(2000);
    expect(r.trace[2000].info.temperature as number).toBeCloseTo(0.912, 3);
  });

  it('every preset names registered methods and an existing problem', () => {
    for (const p of PRESETS) {
      expect(getProblem(p.problem!)).toBeTruthy();
      for (const m of p.methods!) expect(() => getMethod(m.id)).not.toThrow();
      expect(p.methods!.length).toBeLessThanOrEqual(4);
      expect(new Set(p.methods!.map((m) => m.slot)).size).toBe(p.methods!.length);
    }
  });
});

describe('geometry', () => {
  it('phaseAt: whole steps are complete (p = 1); in between, step ⌈t⌉ is being built', () => {
    expect(phaseAt(0, 10)).toEqual({ K: 0, p: 1 });
    expect(phaseAt(3, 10)).toEqual({ K: 3, p: 1 });
    const mid = phaseAt(3.25, 10);
    expect(mid.K).toBe(4);
    expect(mid.p).toBeCloseTo(0.25, 12);
    expect(phaseAt(42, 10)).toEqual({ K: 9, p: 1 });
  });

  it('gaussianEllipseMatrix inverts σ²C; lerpCov stays SPD', () => {
    const C = [
      [2, 0.5],
      [0.5, 1],
    ];
    const M = gaussianEllipseMatrix(0.5, C)!;
    const S = C.map((r) => r.map((v) => v * 0.25));
    const I = [
      [S[0][0] * M[0][0] + S[0][1] * M[1][0], S[0][0] * M[0][1] + S[0][1] * M[1][1]],
      [S[1][0] * M[0][0] + S[1][1] * M[1][0], S[1][0] * M[0][1] + S[1][1] * M[1][1]],
    ];
    expect(I[0][0]).toBeCloseTo(1, 12);
    expect(I[0][1]).toBeCloseTo(0, 12);
    expect(I[1][1]).toBeCloseTo(1, 12);
    expect(
      gaussianEllipseMatrix(1, [
        [1, 1],
        [1, 1],
      ]),
    ).toBeNull();
    const L = lerpCov(
      1,
      C,
      2,
      [
        [1, 0],
        [0, 3],
      ],
      0.5,
    );
    expect(L[0][0] * L[1][1] - L[0][1] ** 2).toBeGreaterThan(0);
  });

  it('deHighlight picks a target whose trial mixes target and mutant coordinates', () => {
    const pop = [
      [0, 0],
      [1, 1],
    ];
    const mut = [
      [5, 5],
      [3, 4],
    ];
    const trials = [
      [5, 5], // all from the mutant
      [3, 1], // x from the mutant, y from the target
    ];
    expect(deHighlight(pop, mut, trials, 0)).toBe(1);
  });

  it('clampToBox and globalMin', () => {
    expect(
      clampToBox(
        [9, -9],
        [
          [-5, 5],
          [-4, 4],
        ],
      ),
    ).toEqual([5, -4]);
    expect(globalMin(getProblem('rastrigin') as Problem<Vector>)).toBe(0);
    expect(globalMin(getProblem('six_hump_camel') as Problem<Vector>)).toBeCloseTo(-1.0316, 4);
  });

  it('track alpha fades with age and stays visible', () => {
    expect(trackAlpha(0)).toBeCloseTo(0.9, 12);
    expect(trackAlpha(TRACK_FADE)).toBeCloseTo(0.16, 12);
    expect(trackAlpha(10 * TRACK_FADE)).toBeCloseTo(0.16, 12);
  });
});

describe('every method, every step: tracks, scales, equations and tables', () => {
  const runs = GLOBAL_METHODS.map((id) => ({ id, r: run(id, 'rastrigin', { max_iter: 30 }, 2) }));

  it('the track is the chain / mean / best as documented', () => {
    for (const { id, r } of runs) {
      for (const s of r.trace) {
        const q = trackPoint(id, s);
        const want =
          id === 'simulated_annealing' || id === 'basin_hopping'
            ? (s.info.current as Vector)
            : id === 'cma_es'
              ? (s.info.mean as Vector)
              : (s.x as Vector);
        expect(q).toEqual([want[0], want[1]]);
      }
    }
  });

  it('search scale and temperature', () => {
    const sa = runs.find((x) => x.id === 'simulated_annealing')!.r;
    expect(searchScale('simulated_annealing', sa.trace[1])).toBeCloseTo(
      Math.max(...(sa.trace[1].info.proposal_sd as Vector)),
      15,
    );
    expect(temperatureOf('simulated_annealing', sa.trace[3])).toBe(sa.trace[3].info.temperature);
    const bh = runs.find((x) => x.id === 'basin_hopping')!.r;
    expect(temperatureOf('basin_hopping', bh.trace[0], 1)).toBeNull();
    expect(temperatureOf('basin_hopping', bh.trace[2], 1)).toBe(1);
    const ps = runs.find((x) => x.id === 'particle_swarm')!.r;
    const s = ps.trace[5];
    const g = s.x as Vector;
    const want = Math.max(
      ...(s.info.particles as Vector[]).map((x) =>
        Math.max(Math.abs(x[0] - g[0]), Math.abs(x[1] - g[1])),
      ),
    );
    expect(searchScale('particle_swarm', s)).toBe(want);
    const de = runs.find((x) => x.id === 'differential_evolution')!.r;
    expect(acceptedCount(de.trace[0])).toBe(0);
    expect(acceptedCount(de.trace[1])).toBe(
      (de.trace[1].info.accepted as boolean[]).filter(Boolean).length,
    );
  });

  it('equations, quantities and tables render every step without NaN', () => {
    for (const { id, r } of runs) {
      const params = { ...defaults(getMethod(id).spec) };
      const cols = iterationColumns(id, 20);
      r.trace.forEach((s: Step, k) => {
        const t = stepTex({ method: id, trace: r.trace, k, params, n: 2 });
        if (k === 0) expect(t).toBeNull();
        else {
          expect(t!.tex).not.toMatch(/NaN|undefined/);
          expect(t!.note.length).toBeGreaterThan(10);
        }
        for (const q of stepQuantities(id, s, 20)) expect(q.value).not.toMatch(/NaN|undefined/);
        for (const c of cols) expect(String(c.value(s))).not.toMatch(/NaN|undefined/);
        const pv = populationView(id, s);
        if (id === 'simulated_annealing') expect(pv).toBeNull();
        else {
          expect(pv).not.toBeNull();
          for (const row of pv!.rows) for (const c of pv!.columns) c.value(row, 0);
        }
      });
    }
  });

  it('the annealing equation is the Metropolis probability of the step', () => {
    const sa = runs.find((x) => x.id === 'simulated_annealing')!.r;
    const k = sa.trace.findIndex(
      (s, i) =>
        i > 0 &&
        s.info.inside === true &&
        (s.info.candidate_f as number) > (sa.trace[i - 1].info.current_f as number),
    );
    expect(k).toBeGreaterThan(0);
    const t = stepTex({ method: 'simulated_annealing', trace: sa.trace, k, params: {}, n: 2 })!;
    expect(t.tex).toContain(tn(sa.trace[k].info.accept_prob as number, 3));
  });

  it('tn typesets numbers like the brand (×10ⁿ outside [10⁻³, 10⁵), trailing zeros kept)', () => {
    expect(tn(0.000123)).toBe('1.230\\times 10^{-4}');
    expect(tn(1e-8)).toBe('1.000\\times 10^{-8}');
    expect(tn(12.3456)).toBe('12.35');
    expect(tn(1.00002)).toBe('1.000');
    expect(tn(99999.9)).toBe('1.000\\times 10^{5}');
    expect(tn(Infinity)).toBe('\\infty');
    expect(tn(null)).toBe('\\text{—}');
    expect(tp(0.7298)).toBe('0.7298');
    expect(tp(1e-8)).toBe('10^{-8}');
    expect(tp(2.5e-6)).toBe('2.5\\times 10^{-6}');
  });
});

/** Parse the numbers of a TeX string (minus signs as '-'). */
const nums = (tex: string) =>
  (tex.match(/-?\d+\.?\d*(?:\\times 10\^\{-?\d+\})?/g) ?? []).map((t) => {
    const m = t.match(/^(-?[\d.]+)(?:\\times 10\^\{(-?\d+)\})?$/)!;
    return Number(m[1]) * 10 ** Number(m[2] ?? 0);
  });

describe('"This step" numbers can be recomputed from what is printed', () => {
  it('Metropolis: no double minus, Δf is the difference of the printed operands, p follows', () => {
    // Styblinski–Tang-like values: both negative, Δf small against |f|.
    const fy = -64.19412345,
      fx = -64.20353111,
      T = 9.979e-4;
    const p = Math.exp(-(fy - fx) / T);
    const tex = metropolisTex(fy, fx, T, p, 'f(\\mathbf{y})', 'T_k');
    expect(tex).not.toMatch(/- -/);
    expect(tex).toContain('(-64.194123)');
    expect(tex).toContain('(-64.203531)');
    expect(tex).toContain('= 0.009408');
    const [d] = tex
      .match(/= (0\.\d+)\\\\/)!
      .slice(1)
      .map(Number);
    expect(d).toBeCloseTo(-64.194123 - -64.203531, 10);
    // The printed exponent reproduces p to the printed digits.
    const ex = Number(tex.match(/= e\^\{-([\d.]+)\}/)![1]);
    expect(Math.abs(Math.exp(-ex) / p - 1)).toBeLessThan(2e-3);
    // Downhill: p = 1, no exponent.
    const down = metropolisTex(-2, -1, 1, 1, 'f(\\mathbf{z})', 'T');
    expect(down).toMatch(/\\le 0\\\\ p_k &= 1\\end\{aligned\}$/);
    expect(down).not.toContain('e^{');
  });

  it('decimalsFor resolves a difference to the asked digits', () => {
    expect(decimalsFor(0.009412, 4)).toBe(6);
    expect(decimalsFor(15.3, 4)).toBe(2);
    expect(decimalsFor(1e-14, 4)).toBe(10);
  });

  it('CMA-ES mean: printed m_k + printed shift = printed m_{k+1}', () => {
    const m0 = [0.99876, 0.99742],
      m1 = [1.0002, 1.00035];
    const tex = meanSum(m0, m1);
    expect(tex).toContain('1.00020');
    expect(tex).toContain('1.00035');
    const v = nums(tex);
    // (a0, a1) + (s0, s1) = (b0, b1)
    expect(v[0] + v[2]).toBeCloseTo(v[4], 10);
    expect(v[1] + v[3]).toBeCloseTo(v[5], 10);
  });

  it('basin hopping: "same minimum" uses the catalog tolerance', () => {
    expect(sameMinimum([2.7468, 2.7468], [2.74681, 2.74679])).toBe(true);
    expect(sameMinimum([2.7468, -2.9], [2.7468, 2.7468])).toBe(false);
  });

  it('basin hopping notes never claim a move to the same minimum, and count in the right number', () => {
    const r = run('basin_hopping', 'styblinski_tang', {});
    const params = defaults(getMethod('basin_hopping').spec);
    for (let k = 1; k < r.trace.length; k++) {
      const s = r.trace[k];
      const note = stepTex({ method: 'basin_hopping', trace: r.trace, k, params, n: 2 })!.note;
      const same = sameMinimum(s.info.local_min as Vector, r.trace[k - 1].info.current as Vector);
      if (s.info.accepted && same) expect(note).toContain('returned to the current minimum');
      if ((s.info.minima as unknown[]).length === 1) expect(note).toContain('1 distinct minimum;');
      expect(note).not.toMatch(/\b1 distinct minima\b|\b1 hops\b/);
    }
  });

  it('the quantity grid has no empty cell', () => {
    for (const wide of [
      [false, false, false, true, false, false, false],
      [false, false, true, false],
      [false, true, false],
    ]) {
      const spans = cellSpans(wide);
      expect(spans.reduce((a, b) => a + b, 0) % 3).toBe(0);
    }
    for (const id of GLOBAL_METHODS) {
      const r = run(id, 'rastrigin', {});
      const q = stepQuantities(id, r.trace[1], 20);
      const spans = cellSpans(q.map((c) => !!c.wide));
      expect(spans.reduce((a, b) => a + b, 0) % 3).toBe(0);
    }
  });
});

describe('DE construction', () => {
  for (const strategy of ['rand/1/bin', 'best/1/bin']) {
    it(`recovers the donors of every mutant (${strategy})`, () => {
      const r = run('differential_evolution', 'rastrigin', { strategy, max_iter: 8 });
      const F = Number(defaults(getMethod('differential_evolution').spec).F);
      for (let k = 1; k < r.trace.length; k++) {
        const old = r.trace[k - 1].info.population as Vector[];
        const best = r.trace[k - 1].info.best as Vector;
        (r.trace[k].info.mutants as Vector[]).forEach((v, i) => {
          const d = deDonors(old, best, v, i, F, strategy)!;
          expect(d).not.toBeNull();
          expect(d.r).not.toContain(i);
          expect(new Set(d.r).size).toBe(d.r.length);
          if (strategy === 'best/1/bin') expect(d.base).toBe(best);
          v.forEach((vj, j) => expect(d.base[j] + F * (d.plus[j] - d.minus[j])).toBe(vj));
        });
      }
    });
  }

  it('the MethodCard rule shows the mutation of the selected strategy', () => {
    expect(deRule('rand/1/bin')).toContain('\\mathbf{x}_{r_1} + F');
    expect(deRule('rand/1/bin')).not.toContain('\\mathbf{x}_{\\mathrm{best}} + F');
    expect(deRule('best/1/bin')).toContain('\\mathbf{x}_{\\mathrm{best}} + F');
    expect(deRule('best/1/bin')).not.toContain('x}_{r_3}');
    const both = getMethod('differential_evolution').doc!.rule!;
    expect(both).toContain('rand/1');
    expect(both).toContain('best/1');
  });

  it('the CMA-ES rule has the δ(h_σ) term, p_c and the σ update', () => {
    const rule = getMethod('cma_es').doc!.rule!;
    expect(rule).toContain('c_1 \\delta(h_\\sigma)');
    expect(rule).toContain('\\mathbf{p}_c &\\leftarrow');
    expect(rule).toContain('\\sigma &\\leftarrow');
  });
});

describe('autoplay waits for the field', () => {
  it('rowVaries: a flat row is the empty surface, a varied row is the field', () => {
    const flat = new Uint8ClampedArray(4 * 50).fill(250);
    expect(rowVaries(flat)).toBe(false);
    const field = flat.slice();
    field[4 * 30] = 120;
    expect(rowVaries(field)).toBe(true);
  });
});

describe('grid values never look more exact than they are', () => {
  it('rounded values keep trailing zeros; exact ones print short', async () => {
    const { show } = await import('../../src/labs/global/stepTex');
    expect(show(3.3)).toBe('3.3');
    expect(show(1.00002)).toBe('1.0000');
    expect(show([1.00002, -2.6])).toBe('(1.0000, −2.6)');
    expect(show(2.9003e-7)).toBe('2.90×10⁻⁷');
    expect(show(1e-8)).toBe('1×10⁻⁸');
  });
});

describe('CMA-ES mean line late in a run', () => {
  it('a tiny shift is printed alone (3 digits), the new mean as ≈', () => {
    const tex = meanSum([-2.9035340252, -2.9035340312], [-2.9035340253, -2.9035340291]);
    expect(tex).toContain('\\mathbf{m}_k + (');
    expect(tex).toContain('\\approx (-2.904,\\ -2.904)');
    expect(tex).not.toMatch(/\d{7,}/);
  });
});
