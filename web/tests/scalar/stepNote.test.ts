/**
 * "This step" notes (src/labs/scalar/stepNote.ts): every equation the note prints must be the one
 * the method evaluated — Fibonacci's last ε-step drops both pieces it drops, Brent's u = x + p/q
 * only when the trial is the vertex, minimum bracketing names both points of a two-point step,
 * and printed points the method keeps apart never print as the same number.
 */
import { describe, expect, it } from 'vitest';
import { getMethod } from '../../src/core/registry';
import type { Result } from '../../src/core/types';
import { getProblem, listProblems } from '../../src/problems/registry';
import type { ScalarProblem } from '../../src/problems/scalar_min';
import '../../src/methods/scalar/methods';
import '../../src/problems/scalar_min';
import { nextView, stepViews } from '../../src/labs/scalar/geometry';
import {
  digitsForPoints,
  droppedEnds,
  isEpsilonStep,
  stepNote,
  texNum,
} from '../../src/labs/scalar/stepNote';
import { presentProblem } from '../../src/labs/scalar/problemCopy';

const PROBLEMS = listProblems<ScalarProblem>('scalar_min');

function run(id: string, problemId: string, params: Record<string, unknown> = {}): Result {
  const { spec, fn } = getMethod(id);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(problemId), { ...defaults, ...(params as Record<string, never>) });
}

/** Numbers printed in a TeX string, in order. */
const numbers = (tex: string) => (tex.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);

describe('Fibonacci: the last (ε) step', () => {
  for (const pid of ['quadratic_1d', 'sin_1d', 'drug_concentration', 'quartic_1d']) {
    it(`${pid}: the note states both cuts and they add up to the next bracket`, () => {
      const r = run('fibonacci_search', pid);
      const v = stepViews('fibonacci_search', r.trace);
      const k = r.trace.findIndex((_, i) => isEpsilonStep('fibonacci_search', r.trace, i + 1));
      expect(k).toBeGreaterThan(0);
      const note = stepNote('fibonacci_search', r, v, k)!;
      expect(note).toContain('q = p + \\varepsilon');
      expect(note.match(/\\text\{drop \}/g)?.length).toBe(2);
      // The two cuts as printed: the first from f(λ) − f(μ), the second from f(p) − f(q).
      const [lam, mu] = v[k].probes;
      const first = lam.f - mu.f > 0 ? 'left' : 'right';
      const nb = nextView('fibonacci_search', r.trace, k)!.nextBracket!;
      const ends = droppedEnds(v[k].bracket!, nb);
      const cutsLeft = /drop \} \[a_\{\d+\},\\ \\lambda\)|drop \} \[a_\{\d+\},\\ p\)/.test(note);
      const cutsRight = /drop \} \(\\mu,\\ b_\{\d+\}\]|drop \} \(q,\\ b_\{\d+\}\]/.test(note);
      expect(ends.left).toBe(cutsLeft || /drop \} \[\\lambda,\\ p\)/.test(note));
      expect(ends.right).toBe(cutsRight || /drop \} \(q,\\ \\mu\]/.test(note));
      expect(first === 'left' ? ends.left : ends.right).toBe(true);
    });
  }

  it('earlier steps cut once, on the side the sign of f(λ) − f(μ) says', () => {
    const r = run('fibonacci_search', 'quadratic_1d');
    const v = stepViews('fibonacci_search', r.trace);
    for (let k = 0; k + 1 < r.trace.length; k++) {
      if (isEpsilonStep('fibonacci_search', r.trace, k + 1)) continue;
      const note = stepNote('fibonacci_search', r, v, k)!;
      expect(note.match(/\\text\{drop \}/g)?.length).toBe(1);
      const [lam, mu] = v[k].probes;
      expect(note).toContain(lam.f - mu.f > 0 ? '> 0' : '\\le 0');
      expect(note).toContain(lam.f - mu.f > 0 ? `[a_{${k}},\\ \\lambda)` : `(\\mu,\\ b_{${k}}]`);
    }
  });
});

describe('Brent: u = x + p/q only when the trial is the vertex', () => {
  for (const pid of ['drug_concentration', 'quartic_1d', 'sin_1d', 'rational_1d']) {
    it(pid, () => {
      const r = run('brent_minimize', pid);
      const v = stepViews('brent_minimize', r.trace);
      let guarded = 0;
      for (let k = 0; k + 1 < r.trace.length; k++) {
        const note = stepNote('brent_minimize', r, v, k)!;
        const nx = r.trace[k + 1].info;
        if (note.includes('u &= x + p/q')) expect(nx.trial).toBe(nx.vertex);
        if (nx.step === 'parabolic' && nx.trial !== nx.vertex) {
          guarded++;
          expect(note).toContain('x \\pm \\text{tol}');
          expect(note).not.toContain('u &= x + p/q');
        }
      }
      // Near convergence Brent always takes minimum steps of length tol.
      if (pid === 'drug_concentration') expect(guarded).toBeGreaterThan(0);
    });
  }
});

describe('minimum bracketing names both points of a two-point step', () => {
  it('parabolic_far: u (the vertex) and u′ = u + φ(u − c)', () => {
    const r = run('bracket_minimum', 'drug_concentration', { x0: 1 });
    const v = stepViews('bracket_minimum', r.trace);
    const nx = nextView('bracket_minimum', r.trace, 0)!;
    expect(nx.stepKind).toBe('parabolic_far');
    expect(nx.trials.length).toBe(2);
    const note = stepNote('bracket_minimum', r, v, 0)!;
    expect(note).toContain("u' = u + \\varphi");
    const printed = numbers(note);
    // Both trial points appear, the vertex first.
    const iu = printed.findIndex((p) => Math.abs(p - nx.trials[0][0]) < 1e-3);
    const iu2 = printed.findIndex((p) => Math.abs(p - nx.trials[1][0]) < 1e-3);
    expect(iu).toBeGreaterThan(-1);
    expect(iu2).toBeGreaterThan(iu);
  });
});

describe('printed digits separate the points', () => {
  it('distinct points print as distinct numbers', () => {
    const xs = [1.3469974, 1.3469975, 2];
    const d = digitsForPoints(xs, 1.347);
    expect(new Set(xs.map((x) => texNum(x, d))).size).toBe(3);
  });

  it('parabolic interpolation on the double well: the triple never prints two equal entries', () => {
    const r = run('parabolic_interpolation', 'quartic_1d');
    const v = stepViews('parabolic_interpolation', r.trace);
    for (let k = 0; k < r.trace.length; k++) {
      const t = r.trace[k].info.triple as number[];
      if (new Set(t).size < 3) continue;
      const note = stepNote('parabolic_interpolation', r, v, k)!;
      const line = note.split(' \\\\ ')[0];
      const printed = line.slice(line.indexOf('&=')).match(/-?\d+(?:\.\d+)?/g)!;
      expect(new Set(printed).size).toBe(3);
    }
  });

  it('the dichotomous rounding stop states the tie', () => {
    const r = run('dichotomous_search', 'drug_concentration', {
      xtol: 1e-10,
      delta_ratio: 0.001,
    });
    const v = stepViews('dichotomous_search', r.trace);
    const k = r.trace.length - 1;
    const [p, q] = v[k].probes;
    if (p.f === q.f) expect(stepNote('dichotomous_search', r, v, k)).toContain('rounding tie');
    // x₁ and x₂ are printed as different numbers even though they are 10⁻¹³ apart.
    const note = stepNote('dichotomous_search', r, v, k)!;
    const row = note.split(' \\\\ ').find((l) => l.startsWith('x_1,\\ x_2'))!;
    const [s1, s2] = row.slice(row.indexOf('&=') + 2).split(',\\ ');
    expect(s1.trim()).not.toBe(s2.trim());
  });
});

describe('rail copy', () => {
  it('keeps id, name and tags; typesets x⋆ and drops developer notes', () => {
    for (const p of PROBLEMS) {
      const s = presentProblem(p);
      expect(s.id).toBe(p.id);
      expect(s.name).toBe(p.name);
      expect(s.tags).toEqual(p.tags);
      expect(s.f).toBe(p.f);
      expect(s.description).not.toMatch(/\*|legacy|nan/);
      expect(s.description).not.toMatch(/f''|f'/);
    }
    expect(presentProblem(getProblem<ScalarProblem>('drug_concentration')).latex).toContain('f(x)');
  });
});
