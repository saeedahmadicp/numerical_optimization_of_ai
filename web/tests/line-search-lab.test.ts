/**
 * Line-search lab: the geometry the lab draws must be the geometry the method used.
 *
 *   - the acceptance tests of geometry.ts agree with every recorded verdict of the Python
 *     fixtures (Step.info.conditions / accepted);
 *   - the zoom interpolant rebuilt from the trace reproduces every strong Wolfe zoom trial
 *     (its minimizer, clamped into the safeguard interval when the method says so);
 *   - φ along 𝐩 reproduces the recorded φ(αₖ); acceptable intervals contain the accepted steps;
 *   - the first local minimizer, the α window, the KaTeX number format;
 *   - every "Try this" preset shows what its title claims.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson } from '../src/core/json';
import { defaults, getMethod } from '../src/core/registry';
import type { Problem2D, Result, Step } from '../src/core/types';
import type { LineSearchKind } from '../src/methods/line_search/methods';
import { getProblem } from '../src/problems/registry';
import '../src/labs/line-search/setup';
import type { LabRun, MethodSelection } from '../src/labs/_shell';
import {
  SAFEGUARD,
  acceptableIntervals,
  accepts,
  armijo,
  chooseWindow,
  goldsteinLower,
  lineMinimizer,
  makeLine,
  sampleLine,
  shortName,
  trialOf,
  zoomInterpolant,
} from '../src/labs/line-search/geometry';
import { buildModel } from '../src/labs/line-search/model';
import { num, texNum, vecNum } from '../src/labs/line-search/format';
import { searchFrame } from '../src/labs/line-search/plane';
import { DEFAULT_SELECTION, PRESETS } from '../src/labs/line-search/config';

const FIX = fileURLToPath(new URL('../src/generated/fixtures/line_search.json', import.meta.url));
const cases = (JSON.parse(readFileSync(FIX, 'utf8')) as Record<string, unknown>[]).map(
  fixtureCaseFromJson,
);

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1, Math.abs(b));

function runOne(problem: Problem2D, sel: MethodSelection, x0: number[], direction: string): LabRun {
  const method = getMethod(sel.id);
  const result = method.fn(problem, {
    ...defaults(method.spec),
    ...sel.params,
    direction,
    x0,
  }) as Result;
  return { sel, method, result } as unknown as LabRun;
}

describe('line-search lab geometry vs the Python fixtures', () => {
  it('has fixtures', () => expect(cases.length).toBeGreaterThanOrEqual(8));

  for (const c of cases) {
    const kind = c.method as LineSearchKind;
    const trace = c.result.trace as Step[];
    const name = `${c.method} on ${c.problem} ${JSON.stringify(c.params)}`;
    const problem = getProblem<Problem2D>(c.problem);
    const t0 = trialOf(trace[0]);
    const x0 = trace[0].x as number[];

    it(`${name}: acceptance tests agree with every recorded verdict`, () => {
      for (const s of trace.slice(1)) {
        const t = trialOf(s);
        expect(armijo(t.alpha, t.phi, t.phi0, t.dphi0, t.c1)).toBe(t.conditions.armijo);
        if (kind === 'goldstein')
          expect(goldsteinLower(t.alpha, t.phi, t.phi0, t.dphi0, t.c1)).toBe(
            t.conditions.goldstein_lower,
          );
        const slopeKnown = t.dphi !== null || kind === 'backtracking' || kind === 'goldstein';
        if (slopeKnown || kind === 'exact_quadratic') {
          const ok = accepts(
            kind,
            t.alpha,
            t.phi,
            t.dphi ?? NaN,
            t.phi0,
            t.dphi0,
            t.c1,
            t.c2 ?? 0.9,
          );
          if (t.accepted) expect(ok).toBe(true);
          else if (slopeKnown) expect(ok).toBe(false);
        }
      }
    });

    it(`${name}: φ along 𝐩 reproduces the recorded φ(αₖ)`, () => {
      const line = makeLine(
        problem,
        x0,
        t0.dphi0 < 0 ? (trace[0].info.direction as number[]) : [0, 0],
      );
      for (const s of trace.slice(1)) {
        const t = trialOf(s);
        if (Number.isFinite(t.phi)) expect(rel(line.phi(t.alpha), t.phi)).toBeLessThan(1e-12);
      }
    });

    if (kind === 'strong_wolfe')
      it(`${name}: the rebuilt interpolant reproduces every zoom trial`, () => {
        let zooms = 0;
        for (let k = 1; k < trace.length; k++) {
          const t = trialOf(trace[k]);
          if (t.phase !== 'zoom') {
            expect(zoomInterpolant(trace, k)).toBeNull();
            continue;
          }
          zooms++;
          const ip = zoomInterpolant(trace, k)!;
          expect(ip).not.toBeNull();
          expect(ip.how).toBe(t.interp);
          const [a, b] = t.interval!;
          expect(ip.safe[0]).toBeCloseTo(a + SAFEGUARD * (b - a), 14);
          if (t.interp === 'cubic' || t.interp === 'quadratic') expect(ip.tStar).toBe(t.alpha);
          else if (t.interp?.endsWith('_clamped'))
            expect(Math.min(Math.max(ip.tStar!, ip.safe[0]), ip.safe[1])).toBe(t.alpha);
          else expect(t.alpha).toBe(a + 0.5 * (b - a));
          // The interpolant passes through the recorded values at both ends.
          const lo =
            t.alphaLo === 0
              ? t.phi0
              : trace.map(trialOf).find((q) => q.k > 0 && q.alpha === t.alphaLo)!.phi;
          expect(rel(ip.value(t.alphaLo!), lo)).toBeLessThan(1e-12);
        }
        if (c.problem !== 'quadratic_ill') expect(zooms).toBeGreaterThan(0);
      });

    if (c.result.converged)
      it(`${name}: the acceptable set contains the accepted step`, () => {
        const p = trace[0].info.direction as number[];
        const line = makeLine(problem, x0, p);
        const a = c.result.extra.alpha as number;
        const s = sampleLine(line, 0, a * 1.6, 600);
        const iv = acceptableIntervals(kind, line, s, t0.phi0, t0.dphi0, t0.c1, t0.c2 ?? 0.9);
        expect(iv.some(([lo, hi]) => lo - 1e-9 * a <= a && a <= hi + 1e-9 * a)).toBe(true);
      });
  }
});

describe('first local minimizer, window, format', () => {
  it('lineMinimizer finds the exact step on a quadratic', () => {
    const problem = getProblem<Problem2D>('quadratic_ill');
    const g = problem.grad([-2, 2]);
    const line = makeLine(problem, [-2, 2], [-g[0], -g[1]]);
    const exact = runOne(
      problem,
      { id: 'exact_quadratic', slot: 0, params: {} },
      [-2, 2],
      'steepest',
    );
    const star = lineMinimizer(line, 1)!;
    expect(rel(star.alpha, exact.result.extra.alpha as number)).toBeLessThan(1e-6);
  });

  it('lineMinimizer takes the first dip, not the deepest one (Ackley)', () => {
    const problem = getProblem<Problem2D>('ackley');
    const g = problem.grad([2.6, -3.4]);
    const line = makeLine(problem, [2.6, -3.4], [-g[0], -g[1]]);
    const star = lineMinimizer(line, 1000)!;
    expect(star.alpha).toBeGreaterThan(0.2);
    expect(star.alpha).toBeLessThan(0.25);
    expect(Math.abs(line.dphi(star.alpha))).toBeLessThan(1e-5);
  });

  it('lineMinimizer is null when φ decreases on the whole range', () => {
    const line = makeLine({ f: (x) => -x[0], grad: () => [-1, 0] }, [0, 0], [1, 0]);
    expect(lineMinimizer(line, 10)).toBeNull();
  });

  it('chooseWindow: near shows the accepted steps, all shows every trial', () => {
    const trials = [
      [1, 0.5, 0.25, 0.125],
      [1, 0.1, 0.19, 0.1189],
    ];
    const near = chooseWindow(trials, [0.125, 0.1189], 0.127, 'near', 1);
    expect(near).toBeGreaterThan(0.19);
    expect(near).toBeLessThan(0.5);
    expect(chooseWindow(trials, [0.125, 0.1189], 0.127, 'all', 1)).toBeGreaterThanOrEqual(1);
    // A minimizer far beyond every accepted step does not stretch the window.
    expect(chooseWindow([[0.2]], [0.2], 50, 'near', 1)).toBeLessThan(1);
    expect(chooseWindow([[]], [null], null, 'near', 3)).toBe(3);
  });

  it('texNum typesets numbers for KaTeX', () => {
    expect(texNum(136.2)).toBe('136.2');
    expect(texNum(-0.008084)).toBe('-0.008084');
    expect(texNum(-9.35e-5)).toBe('-9.35\\times10^{-5}');
    expect(texNum(1e-12)).toBe('10^{-12}');
    expect(texNum(null)).toBe('\\text{—}');
    expect(texNum(Infinity)).toBe('\\infty');
  });

  it('shortName drops the variant', () => {
    expect(shortName('Strong Wolfe (bracket + zoom)')).toBe('Strong Wolfe');
    expect(shortName('Goldstein')).toBe('Goldstein');
  });
});

describe('the default view and the presets show what they claim', () => {
  const view = (
    problemId: string,
    x0: number[],
    sel: readonly MethodSelection[],
    d = 'steepest',
  ) => {
    const problem = getProblem<Problem2D>(problemId);
    const runs = sel.map((s) => runOne(problem, s, x0, d));
    return { runs, model: buildModel(problem, x0, runs, 'near') };
  };
  const inside = (iv: [number, number][], a: number) => iv.some(([lo, hi]) => lo <= a && a <= hi);

  it('default (Himmelblau): every method converges; Goldstein excludes α⋆', () => {
    const { runs, model } = view('himmelblau', [0, 0], DEFAULT_SELECTION);
    expect(runs.every((r) => r.result.converged)).toBe(true);
    expect(model.alphaStar!.alpha).toBeCloseTo(0.1274, 3);
    const gi = runs.findIndex((r) => r.sel.id === 'goldstein');
    expect(inside(model.intervals[gi], model.alphaStar!.alpha)).toBe(false);
    expect(inside(model.intervals[gi], runs[gi].result.extra.alpha as number)).toBe(true);
    // The window shows every accepted step and α⋆.
    for (const r of runs) expect(r.result.extra.alpha as number).toBeLessThan(model.hi);
  });

  for (const p of PRESETS) {
    it(`preset "${p.title}"`, () => {
      const { runs, model } = view(p.problem!, [...(p.start as number[])], p.methods!, p.extra!.d);
      expect(runs.every((r) => r.result.converged)).toBe(true);
      const by = (id: string) => runs.find((r) => r.sel.id === id)!;
      if (p.id === 'goldstein-excludes') {
        const gi = runs.findIndex((r) => r.sel.id === 'goldstein');
        expect(inside(model.intervals[gi], model.alphaStar!.alpha)).toBe(false);
      }
      if (p.id === 'short-step') {
        expect(by('backtracking').result.nIter).toBe(1);
        expect(by('backtracking').result.extra.alpha).toBe(0.001);
        const sw = by('strong_wolfe').result;
        expect(rel(sw.extra.alpha as number, model.alphaStar!.alpha)).toBeLessThan(1e-3);
        // The note: doubling stops because φ rose, then one quadratic fit.
        const ts = sw.trace.map(trialOf);
        const ex = ts.filter((t) => t.phase === 'expand');
        expect(ex[ex.length - 1].phi).toBeGreaterThanOrEqual(ex[ex.length - 2].phi);
        expect(ts.filter((t) => t.phase === 'zoom').map((t) => t.interp)).toEqual(['quadratic']);
      }
      if (p.id === 'cubic-zoom') {
        const sw = by('strong_wolfe').result.trace.map(trialOf);
        const ex = sw.filter((t) => t.phase === 'expand');
        expect(ex.length).toBe(9);
        // Eight doublings: α₉ = 2⁸ α₁.
        expect(ex[8].alpha / ex[0].alpha).toBe(256);
        expect(sw.filter((t) => t.interp?.startsWith('cubic')).length).toBe(2);
        const ww = by('weak_wolfe').result.trace.map(trialOf);
        expect(ww[ww.length - 1].dphi!).toBeGreaterThan(0);
        expect(ww[ww.length - 1].alpha).toBeGreaterThan(model.alphaStar!.alpha);
      }
      if (p.id === 'newton-unit') {
        for (const r of runs) {
          expect(r.result.nIter).toBe(1);
          expect(rel(r.result.extra.alpha as number, 1)).toBeLessThan(1e-12);
        }
      }
    });
  }
});

describe('plain-digit numbers and the search frame', () => {
  it('num prints plain digits below 10⁵ and ×10ⁿ outside [10⁻³, 10⁵)', () => {
    expect(num(1000, 3)).toBe('1000');
    expect(num(17042, 4)).toBe('17040');
    expect(num(-34240.7, 4)).toBe('−34240');
    expect(num(99999, 3)).toBe('1×10⁵');
    expect(num(0.11888, 4)).toBe('0.1189');
    expect(num(2.5, 4)).toBe('2.5');
    expect(num(-0.5, 4)).toBe('−0.5');
    expect(num(1.5e5, 3)).toBe('1.5×10⁵');
    expect(num(2e-4, 3)).toBe('2×10⁻⁴');
    expect(num(0, 3)).toBe('0');
    expect(num(Infinity, 3)).toBe('∞');
    expect(num(null)).toBe('—');
    expect(vecNum([1000, -2.5])).toBe('(1000, −2.5)');
  });

  it('searchFrame frames a short segment and leaves a long one to the domain', () => {
    const dom: [[number, number], [number, number]] = [
      [-5, 5],
      [-5, 5],
    ];
    // Segment of length 1 (10 % of the domain; the threshold is 25 %): framed, centered on its midpoint.
    const f = searchFrame(dom, [1, 2], [1, 0], 1)!;
    expect(f).not.toBeNull();
    expect((f[0][0] + f[0][1]) / 2).toBeCloseTo(1.5, 12);
    expect((f[1][0] + f[1][1]) / 2).toBeCloseTo(2, 12);
    expect(f[0][1] - f[0][0]).toBeCloseTo(1 / 0.6, 12);
    // The whole segment lies inside the frame.
    expect(f[0][0]).toBeLessThan(1);
    expect(f[0][1]).toBeGreaterThan(2);
    // Very short segments get at least 5 % of the domain.
    const g = searchFrame(dom, [0, 0], [0, 1], 0.01)!;
    expect(g[1][1] - g[1][0]).toBeCloseTo(0.5, 12);
    // Long, empty or non-finite segments are not framed.
    expect(searchFrame(dom, [0, 0], [3, 0], 1)).toBeNull();
    expect(searchFrame(dom, [0, 0], [0, 0], 1)).toBeNull();
    expect(searchFrame(dom, [0, 0], null, 1)).toBeNull();
    expect(searchFrame(dom, [0, 0], [NaN, 1], 1)).toBeNull();
  });

  it('the Ackley and Newton presets get a framed plane; the default view does not', () => {
    const frameOf = (
      problemId: string,
      x0: number[],
      sel: readonly MethodSelection[],
      d: string,
    ) => {
      const problem = getProblem<Problem2D>(problemId);
      const runs = sel.map((s) => runOne(problem, s, x0, d));
      const model = buildModel(problem, x0, runs, 'near');
      return searchFrame(
        problem.domain as [[number, number], [number, number]],
        x0,
        model.p,
        model.hi,
      );
    };
    expect(frameOf('himmelblau', [0, 0], DEFAULT_SELECTION, 'steepest')).toBeNull();
    for (const id of ['cubic-zoom', 'newton-unit']) {
      const p = PRESETS.find((q) => q.id === id)!;
      expect(
        frameOf(p.problem!, [...(p.start as number[])], p.methods!, p.extra!.d as string),
      ).not.toBeNull();
    }
  });
});
