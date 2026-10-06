/** The site pages' pure helpers: method facets, the live parity check, run status wording. */
import { describe, expect, it } from 'vitest';
import { readMethods, readParityCases } from '../vite/catalog';
import { renderStudy, slugify, studyIds } from '../vite/research';
import { describeResult, divergenceReason } from '../src/labs/_shell/status';
import { matchesFilter, needsClass, pyRepr, rateClass } from '../src/site/methodMeta';
import { compareCase, scaledError } from '../src/site/parity';
import type { Result } from '../src/core/types';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const RESEARCH = fileURLToPath(new URL('../../research', import.meta.url));

describe('method facets', () => {
  const methods = readMethods();
  it('classifies every registry rate', () => {
    expect(rateClass('quadratic (simple root); linear at a multiple root')).toBe('quadratic');
    expect(rateClass('cubic (simple root)')).toBe('quadratic');
    expect(rateClass('superlinear (≈ 1.84)')).toBe('superlinear');
    expect(rateClass('linear (rate ½)')).toBe('linear');
    expect(rateClass('sublinear (stochastic)')).toBe('sublinear');
    expect(rateClass('direct, n³/3 flops')).toBe('finite');
    expect(rateClass('O(h⁴); exact for degree ≤ 3')).toBe('truncation');
    expect(rateClass('')).toBe('none');
    // Every method lands in some class, and most have a real rate.
    const none = methods.filter((m) => rateClass(m.order) === 'none');
    expect(none.length).toBeLessThan(methods.length / 4);
  });
  it('classifies derivative needs', () => {
    expect(needsClass(['f', 'grad', 'hess'])).toBe('hess');
    expect(needsClass(['f', 'grad'])).toBe('grad');
    expect(needsClass(['f', 'bracket'])).toBe('f');
    expect(needsClass(['A', 'b'])).toBe('other');
  });
  it('filters by facets and by text over names, ids, sources and parameters', () => {
    const f = { q: '', family: '', needs: '', rate: '', det: '' };
    expect(methods.filter((m) => matchesFilter(m, f)).length).toBe(methods.length);
    const bfgs = methods.find((m) => m.id === 'bfgs')!;
    expect(matchesFilter(bfgs, { ...f, q: 'nocedal 6.1' })).toBe(true);
    expect(matchesFilter(bfgs, { ...f, q: 'gtol' })).toBe(true);
    expect(matchesFilter(bfgs, { ...f, needs: 'hess' })).toBe(false);
    expect(matchesFilter(bfgs, { ...f, det: 'stochastic' })).toBe(false);
    expect(
      methods.filter((m) => matchesFilter(m, { ...f, det: 'stochastic' })).length,
    ).toBeGreaterThan(5);
  });
  it('prints Python reprs', () => {
    expect(pyRepr(1e-8, 'float')).toBe('1e-08');
    expect(pyRepr(0, 'float')).toBe('0.0');
    expect(pyRepr(500, 'int')).toBe('500');
    expect(pyRepr(true)).toBe('True');
    expect(pyRepr('strong_wolfe')).toBe('"strong_wolfe"');
  });
});

describe('build-time parity record', () => {
  const cases = readParityCases();
  it('keeps every fixture case with its first iterates, and counts cases per method', () => {
    expect(cases.length).toBeGreaterThan(100);
    for (const c of cases) {
      expect(c.head.length).toBe(Math.min(10, c.steps));
      expect(typeof c.nIter).toBe('number');
    }
    const methods = readMethods();
    expect(methods.reduce((n, m) => n + m.cases, 0)).toBe(cases.length);
  });
  it('compares like the harness', () => {
    expect(scaledError([1, 2], [1, 2 + 1e-9], 1e-8)).toBeLessThan(1);
    expect(scaledError([1, 2], [1, 2.1], 1e-8)).toBeGreaterThan(1);
    expect(scaledError([1], [1, 2], 1e-8)).toBe(Infinity);
    expect(scaledError(Infinity, Infinity, 1e-8)).toBe(0);
    const c = cases.find((x) => x.method === 'bisection')!;
    const trace = (c.head as number[]).map((x, k) => ({
      k,
      x,
      fun: 0,
      gradNorm: null,
      stepSize: null,
      info: {},
    }));
    const result: Result = {
      method: 'bisection',
      x: c.x,
      fun: 0,
      converged: c.converged,
      message: '',
      nIter: c.nIter,
      nFev: 0,
      nGev: 0,
      nHev: 0,
      trace,
      extra: {},
    };
    expect(compareCase(result, c, true).ok).toBe(true);
    expect(compareCase({ ...result, nIter: c.nIter + 1 }, c, true).ok).toBe(false);
  });
});

describe('run status', () => {
  const base: Result = {
    method: 'm',
    x: 0,
    fun: 0,
    converged: false,
    message: '',
    nIter: 7,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace: [],
    extra: {},
  };
  it('names the cause of a divergence', () => {
    expect(divergenceReason('diverged: |x| = 2.38e+13 > 1.5e+12')).toBe(
      '|x| grew past the divergence bound',
    );
    expect(divergenceReason('diverged: ‖x‖₂ > 1e+08')).toBe('|x| grew past the divergence bound');
    expect(divergenceReason('diverged: non-finite value (x = nan, f = nan)')).toBe(
      'non-finite value',
    );
    expect(divergenceReason('f or ∇f is not finite at iteration 3: the iterates diverged')).toBe(
      'non-finite value',
    );
    expect(divergenceReason('max_iter reached')).toBe(null);
    expect(describeResult({ ...base, message: 'diverged: |x| = 2.38e+13 > 1.5e+12' }).long).toBe(
      'Diverged after 7 iterations (|x| grew past the divergence bound)',
    );
  });
  it('reports cycles and stalls in their own words', () => {
    expect(describeResult({ ...base, message: 'cycle of period 2 detected' }).short).toBe(
      'cycled · 7 iterations',
    );
    expect(
      describeResult({ ...base, message: 'step lengths below 1e-12: the method stalled' }).short,
    ).toBe('stalled · 7 iterations');
  });
});

describe('research rendering', () => {
  const ids = studyIds(RESEARCH);
  it('finds the studies', () => {
    expect(ids.length).toBeGreaterThan(3);
  });
  it('typesets math, rewrites links, collects figures and builds a TOC', () => {
    const md = [
      '# Title with $x$',
      '',
      'Lede with $\\alpha_k$.',
      '',
      '## Question',
      '',
      'See [the other study](../other/) and [method.py](method.py) and [below](#question).',
      '',
      '$$f(x) = x^2$$',
      '',
      '![A figure](figures/missing.svg)',
      '',
      '| a | b |',
      '|---|---|',
      '| $1$ | 2 |',
    ].join('\n');
    const d = renderStudy(md, {
      id: 'demo',
      root: RESEARCH,
      blob: 'https://github.com/o/r/blob/main',
      studies: ['demo', 'other'],
    });
    expect(d.title).toBe('Title with x');
    expect(d.lede).toContain('katex');
    expect(d.html).not.toContain('Lede with');
    expect(d.toc).toEqual([{ depth: 2, text: 'Question', slug: 'question' }]);
    expect(d.html).toContain('href="#/research/other"');
    expect(d.html).toContain('https://github.com/o/r/blob/main/research/demo/method.py');
    expect(d.html).toContain('href="#/research/demo?s=question"');
    expect(d.html).toContain('math-display');
    expect(d.missing).toEqual(['figures/missing.svg']);
    expect(d.html).toContain('table-wrap');
  });
  it('typesets inline math that wraps across a line break, but not across a paragraph', () => {
    const ctx = { id: 'demo', root: RESEARCH, blob: 'https://x/blob/main', studies: [] };
    const d = renderStudy('# T\n\nLede.\n\n## A\n\nSo $g(x_i) = x_i -\n\\alpha$ holds.\n', ctx);
    expect(d.html).toContain('katex');
    expect(d.html).not.toContain('$g(x_i)');
    const p = renderStudy('# T\n\nLede.\n\n## A\n\nCosts $5 and\n\nthen $6.\n', ctx);
    expect(p.html).not.toContain('katex');
  });
  it('leaves no raw TeX in any study', () => {
    for (const id of ids) {
      const d = renderStudy(readFileSync(join(RESEARCH, id, 'README.md'), 'utf8'), {
        id,
        root: RESEARCH,
        blob: 'https://github.com/o/r/blob/main',
        studies: ids,
      });
      const text = (d.lede + d.html)
        .replace(/<annotation[\s\S]*?<\/annotation>/g, '')
        .replace(/<pre[\s\S]*?<\/pre>/g, '')
        .replace(/<code[\s\S]*?<\/code>/g, '')
        .replace(/<[^>]+>/g, '');
      expect(text.match(/\$[^$]*\\[a-zA-Z]/)?.[0], id).toBeUndefined();
    }
  });
  it('slugs headings like GitHub', () => {
    expect(slugify('How benchmarking works')).toBe('how-benchmarking-works');
    expect(slugify('Method (`method.py`)')).toBe('method-methodpy');
  });
});
