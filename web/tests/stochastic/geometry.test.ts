/**
 * The lab's step geometry (src/labs/stochastic/geometry.ts) is exact, not illustrative:
 * - the per-sample chain sums to the recorded update for every method that has one;
 * - the expected step of momentum/Nesterov/SVRG uses the right push and evaluation point;
 * - the covariance equals the empirical covariance of the mini-batch step over random batches;
 * - the filled-in update rule (equation.ts) reproduces the recorded iterate.
 */
import { describe, expect, it } from 'vitest';
import { getMethod, defaults } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import { Rng } from '../../src/core/rng';
import type { FiniteSumProblem } from '../../src/problems/stochastic';
import type { Step } from '../../src/core/types';
import '../../src/methods/stochastic/methods';
import '../../src/problems/stochastic';
import {
  batchFactor,
  covariance,
  dataMode,
  epochOf,
  inverse2,
  ruleKind,
  stepGeometry,
} from '../../src/labs/stochastic/geometry';
import { stepTex, texNum } from '../../src/labs/stochastic/equation';
import { stepOverlays } from '../../src/labs/stochastic/overlays';

const P = (id: string) => getProblem<FiniteSumProblem>(id);
function run(method: string, problem: string, params: Record<string, unknown> = {}) {
  const { spec, fn } = getMethod(method);
  return fn(P(problem), { ...defaults(spec), ...(params as Record<string, never>) });
}

const close = (a: readonly number[], b: readonly number[], tol: number) =>
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol * (1 + Math.abs(b[i]))));

describe('stochastic lab geometry', () => {
  const CHAIN = [
    'sgd',
    'sgd_momentum',
    'sgd_nesterov',
    'svrg',
    'stochastic_adagrad',
    'stochastic_rmsprop',
  ];
  for (const id of CHAIN) {
    it(`${id}: the per-sample chain sums to the recorded update`, () => {
      const r = run(id, 'logreg_2d', { batch_size: 7, epochs: 2, lr: 0.05 });
      for (const s of r.trace.slice(1, 20)) {
        const g = stepGeometry(P('logreg_2d'), id, s, 7)!;
        expect(g.chain).not.toBeNull();
        const last = g.chain![g.chain!.length - 1];
        close(last, g.to, 1e-12);
        // One part per sample (SVRG adds the snapshot gradient as its first part).
        expect(g.chain!.length).toBe((s.info.batch as number[]).length + (id === 'svrg' ? 2 : 1));
      }
    });
  }

  it('Adam, SAG and SAGA have no per-sample chain; b > 32 has none either', () => {
    for (const id of ['stochastic_adam', 'sag', 'saga']) {
      const s = run(id, 'linreg_2d', { batch_size: 5, epochs: 1 }).trace[3];
      expect(stepGeometry(P('linreg_2d'), id, s, 5)!.chain).toBeNull();
    }
    const s = run('sgd', 'linreg_2d', { batch_size: 40, epochs: 1 }).trace[2];
    expect(stepGeometry(P('linreg_2d'), 'sgd', s, 40)!.chain).toBeNull();
  });

  it('momentum: push = βv_{k−1}, and the expected step starts there', () => {
    const r = run('sgd_momentum', 'huber_regression_2d', { batch_size: 10, epochs: 1 });
    for (let i = 2; i < 12; i++) {
      const s = r.trace[i];
      const g = stepGeometry(P('huber_regression_2d'), 'sgd_momentum', s, 10)!;
      const vPrev = r.trace[i - 1].info.velocity as number[]; // every = 1 here
      close(
        [g.pushTo[0] - g.base[0], g.pushTo[1] - g.base[1]],
        [0.9 * vPrev[0], 0.9 * vPrev[1]],
        1e-12,
      );
      const full = P('huber_regression_2d').grad(g.base);
      close(g.mean!, [g.pushTo[0] - 0.02 * full[0], g.pushTo[1] - 0.02 * full[1]], 1e-12);
    }
  });

  it('Nesterov evaluates at the recorded look-ahead point', () => {
    const r = run('sgd_nesterov', 'linreg_2d', { batch_size: 10, epochs: 1 });
    for (const s of r.trace.slice(1, 10)) {
      const g = stepGeometry(P('linreg_2d'), 'sgd_nesterov', s, 10)!;
      close(g.evalAt, s.info.lookahead as number[], 1e-12);
    }
  });

  it('the covariance is the covariance of the mini-batch step (Monte Carlo, 20,000 batches)', () => {
    const p = P('linreg_2d');
    const s = run('sgd', 'linreg_2d', { batch_size: 8, epochs: 1, lr: 0.1 }).trace[3];
    const g = stepGeometry(p, 'sgd', s, 8)!;
    const rng = new Rng(7);
    const steps: number[][] = [];
    for (let t = 0; t < 20_000; t++) {
      const batch = rng.permutation(p.nSamples).slice(0, 8);
      const gb = p.gradBatch(g.evalAt, batch);
      steps.push([-0.1 * gb[0], -0.1 * gb[1]]);
    }
    const C = covariance(steps);
    for (const [i, j] of [
      [0, 0],
      [0, 1],
      [1, 1],
    ])
      expect(Math.abs(C[i][j] - g.cov![i][j])).toBeLessThan(
        0.05 * Math.sqrt(g.cov![i][i] * g.cov![j][j]),
      );
    // The mean of the step is −η∇f.
    const m = [0, 1].map((c) => steps.reduce((a, v) => a + v[c], 0) / steps.length);
    const full = p.grad(g.evalAt);
    close(m, [-0.1 * full[0], -0.1 * full[1]], 0.02);
  });

  it('SVRG covariance vanishes when the iterate equals the snapshot', () => {
    const r = run('svrg', 'linreg_2d', { batch_size: 10, epochs: 2 });
    // The first update of an epoch starts at the snapshot: the corrections cancel exactly.
    const s = r.trace.find((st) => st.k === 21)!;
    const g = stepGeometry(P('linreg_2d'), 'svrg', s, 10)!;
    close(g.base, s.info.snapshot as number[], 1e-12);
    expect(Math.abs(g.cov![0][0]) + Math.abs(g.cov![1][1])).toBeLessThan(1e-28);
    expect(g.noise).toBeLessThan(1e-12);
  });

  it('the last mini-batch of an epoch has its own size; full batches have no ellipse', () => {
    expect(batchFactor(200, 200)).toBe(0);
    expect(batchFactor(200, 1)).toBeCloseTo(1, 12);
    const r = run('sgd', 'linreg_2d', { batch_size: 30, epochs: 1 }); // U = 7, last batch 20
    const s = r.trace[7];
    expect((s.info.batch as number[]).length).toBe(20);
    const g = stepGeometry(P('linreg_2d'), 'sgd', s, 30)!;
    const S = covariance(P('linreg_2d').gradSamples(g.evalAt, [...Array(200).keys()]));
    const f = (0.05 * 0.05 * (200 - 20)) / (20 * 199);
    close([g.cov![0][0], g.cov![1][1]], [f * S[0][0], f * S[1][1]], 1e-12);
    const full = run('sgd', 'linreg_2d', { batch_size: 500, epochs: 1 }).trace[1];
    expect(stepGeometry(P('linreg_2d'), 'sgd', full, 500)!.cov).toBeNull();
  });

  it('k = 0 has no geometry; helpers', () => {
    const s0 = run('sgd', 'linreg_2d', { epochs: 1 }).trace[0];
    expect(stepGeometry(P('linreg_2d'), 'sgd', s0, 10)).toBeNull();
    expect(
      inverse2([
        [4, 0],
        [0, 2],
      ]),
    ).toEqual([
      [0.25, -0],
      [-0, 0.5],
    ]);
    expect(
      inverse2([
        [1, 1],
        [1, 1],
      ]),
    ).toBeNull();
    expect(ruleKind('saga')).toBe('table');
    expect(ruleKind('stochastic_adam')).toBe('adaptive');
    expect(epochOf(30, 20)).toBe(1.5);
    expect(dataMode(P('linreg_2d'))).toBe('line');
    expect(dataMode(P('huber_regression_2d'))).toBe('line');
    expect(dataMode(P('logreg_2d'))).toBe('logistic');
    expect(dataMode(P('ill_conditioned_ls'))).toBe('predicted');
  });

  it('overlays: an ellipse for SGD, a look-ahead for Nesterov, labels only when asked', () => {
    const s = run('sgd_nesterov', 'linreg_2d', { batch_size: 5, epochs: 1 }).trace[2];
    const g = stepGeometry(P('linreg_2d'), 'sgd_nesterov', s, 5)!;
    const lens = stepOverlays(g, 1, { labels: true, chain: true, minLength: 0 });
    expect(lens.some((o) => o.kind === 'ellipse')).toBe(true);
    expect(lens.some((o) => o.kind === 'point' && o.label !== undefined)).toBe(true);
    const main = stepOverlays(g, 1, { labels: false, chain: false, minLength: 0 });
    expect(main.every((o) => !('label' in o) || o.label === undefined)).toBe(true);
    expect(main.some((o) => o.kind === 'polyline')).toBe(false);
  });
});

describe('the filled-in update rule', () => {
  it('formats numbers for KaTeX', () => {
    expect(texNum(0.1)).toBe('0.1');
    expect(texNum(-1.23456)).toBe('-1.235');
    expect(texNum(1.244e-4)).toBe('1.244\\times10^{-4}');
    expect(texNum(4.046e6)).toBe('4.046\\times10^{6}');
    expect(texNum(0)).toBe('0');
  });

  const parseVecs = (tex: string) =>
    [...tex.matchAll(/\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}/g)].map((m) =>
      m[1].split('\\\\').map((t) => Number(t.replace('\\times10^{', 'e').replace('}', ''))),
    );

  for (const id of [
    'sgd',
    'sgd_momentum',
    'sgd_nesterov',
    'stochastic_adagrad',
    'stochastic_rmsprop',
    'stochastic_adam',
    'svrg',
    'saga',
    'sag',
  ]) {
    it(`${id}: the printed numbers reproduce 𝐰ₖ to four digits`, () => {
      const r = run(id, 'linreg_2d', { batch_size: 10, epochs: 1, lr: 0.02 });
      const s: Step = r.trace[5];
      const tex = stepTex(id, s)!;
      expect(tex.startsWith(`\\mathbf{w}_{${s.k}} = `)).toBe(true);
      const vecs = parseVecs(tex);
      const out = vecs[vecs.length - 1];
      close(out, s.x as number[], 1e-3);
      const base = vecs[0];
      const kind = ruleKind(id);
      let pred: number[];
      if (kind === 'momentum' || kind === 'nesterov')
        pred = [0, 1].map((c) => base[c] + vecs[1][c] - s.stepSize! * vecs[2][c]);
      else if (kind === 'adaptive') pred = [0, 1].map((c) => base[c] - vecs[1][c] * vecs[2][c]);
      else pred = [0, 1].map((c) => base[c] - s.stepSize! * vecs[1][c]);
      close(pred, s.x as number[], 2e-3);
    });
  }

  it('is null at k = 0', () => {
    expect(stepTex('sgd', run('sgd', 'linreg_2d', { epochs: 1 }).trace[0])).toBeNull();
  });
});
