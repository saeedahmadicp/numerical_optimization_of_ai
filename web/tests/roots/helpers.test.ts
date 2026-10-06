/**
 * Unit tests for what the parity fixtures do not cover: the correctly rounded libm, Python
 * float semantics (repr, format(·, "g"), nextafter, ulp, ldexp), ITP's exact n½, and
 * invariants of the bracketing methods (the bracket always contains a sign change).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { crAtan, crCos, crExp, crHypot, crSin } from '../../src/methods/roots/crmath';
import {
  copysign,
  fmtG,
  itpNHalf,
  ldexp,
  nextafter,
  pyMax,
  pyMin,
  pyRepr,
  ulp,
} from '../../src/methods/roots/bracketing';
import { stepTestEstimate } from '../../src/methods/roots/open';
import { getMethod, listMethods } from '../../src/core/registry';
import { getProblem, listProblems } from '../../src/problems/registry';
import type { RootProblem } from '../../src/problems/roots';
import '../../src/problems/roots';

const FILE = fileURLToPath(new URL('./fixtures/roots_oracle.json.gz', import.meta.url));
const data = JSON.parse(gunzipSync(readFileSync(FILE)).toString('utf8')) as {
  libm: Record<'x' | 'y' | 'exp' | 'sin' | 'cos' | 'atan' | 'hypot', number[]>;
  format: Record<'repr' | 'g3' | 'g6', [number, string][]>;
  ulp: [number, number, number, number][];
  itp_n_half: [number, number, number, number][];
};

describe('correctly rounded libm agrees with glibc', () => {
  const { x, y } = data.libm;
  const cases: [string, (i: number) => number, number][] = [
    // [name, ours, allowed mismatch rate] — glibc itself is not correctly rounded on ~0.1 %.
    ['exp', (i) => crExp(x[i]), 0.003],
    ['sin', (i) => crSin(x[i]), 0.003],
    ['cos', (i) => crCos(x[i]), 0.003],
    ['atan', (i) => crAtan(x[i]), 0.003],
    ['hypot', (i) => crHypot(x[i], y[i]), 0],
  ];
  for (const [name, fn, rate] of cases) {
    it(`${name}: within one ulp everywhere, equal on all but ${rate * 100} %`, () => {
      const want = data.libm[name as 'exp'];
      let bad = 0;
      want.forEach((w, i) => {
        const v = fn(i);
        if (v !== w) bad++;
        expect(Math.abs(v - w)).toBeLessThanOrEqual(ulp(w));
      });
      expect(bad / want.length).toBeLessThanOrEqual(rate);
    });
  }
  it('handles special values like libm', () => {
    expect(crExp(1000)).toBe(Infinity);
    expect(crExp(-1000)).toBe(0);
    expect(crExp(0)).toBe(1);
    expect(crExp(NaN)).toBeNaN();
    expect(crSin(Infinity)).toBeNaN();
    expect(crCos(NaN)).toBeNaN();
    expect(Object.is(crSin(-0), -0)).toBe(true);
    expect(crAtan(Infinity)).toBe(Math.PI / 2);
    expect(crAtan(-1e300)).toBe(-Math.PI / 2);
    expect(crHypot(3, 4)).toBe(5);
    expect(crHypot(Infinity, NaN)).toBe(Infinity);
    expect(crHypot(1e300, 1e300)).toBe(Math.SQRT2 * 1e300);
  });
});

describe('Python float semantics', () => {
  it('repr', () => {
    for (const [v, s] of data.format.repr) expect(pyRepr(v)).toBe(s);
  });
  it('format(v, ".3g") and ".6g"', () => {
    for (const [v, s] of data.format.g3) expect(fmtG(v, 3)).toBe(s);
    for (const [v, s] of data.format.g6) expect(fmtG(v, 6)).toBe(s);
  });
  it('ulp and nextafter', () => {
    const num = (v: number | string) => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v);
    for (const [v, u, up, down] of data.ulp.map((r) => r.map(num) as number[])) {
      expect(ulp(v)).toBe(u);
      expect(nextafter(v, Infinity)).toBe(up);
      expect(nextafter(v, -Infinity)).toBe(down);
    }
  });
  it('ldexp, copysign, min/max with NaN', () => {
    expect(ldexp(1, 1100)).toBe(Infinity);
    expect(ldexp(1e-300, 1100)).toBe(1e-300 * 2 ** 1000 * 2 ** 100);
    expect(ldexp(1, -1074)).toBe(5e-324);
    expect(copysign(2, -0)).toBe(-2);
    // Python: max(t, nan) = t, max(nan, t) = nan.
    expect(pyMax(0.25, NaN)).toBe(0.25);
    expect(pyMax(NaN, 0.25)).toBeNaN();
    expect(pyMin(0.75, NaN)).toBe(0.75);
  });
  it('ITP n½ is the exact least n with 2ε·2ⁿ ≥ b − a', () => {
    for (const [a, b, eps, n] of data.itp_n_half) expect(itpNHalf(a, b, eps)).toBe(n);
  });
  it('the a posteriori step estimate needs two consecutive decreases', () => {
    expect(stepTestEstimate([0, 1, 1.5])).toBeNull();
    expect(stepTestEstimate([0, 1, 1.2, 1.7])).toBeNull(); // the last step grew
    expect(stepTestEstimate([0, 1, 1.5, 1.75])).toEqual([0.25, 0.5]);
  });
});

describe('bracketing invariants', () => {
  const bracketing = listMethods('roots').filter((m) => m.spec.needs.includes('bracket'));
  const problems = listProblems<RootProblem>('roots');
  it('registers the nine bracketing and seven open methods', () => {
    expect(bracketing.map((m) => m.spec.id).sort()).toEqual(
      [
        'anderson_bjorck',
        'bisection',
        'brent',
        'chandrupatla',
        'illinois',
        'itp',
        'pegasus',
        'regula_falsi',
        'ridders',
      ].sort(),
    );
    expect(listMethods('roots').length).toBe(16);
  });
  for (const m of bracketing) {
    it(`${m.spec.id}: every bracket holds a sign change and x_k; the run ends at a root`, () => {
      for (const p of problems) {
        // Random sub-brackets of the default one that still change sign.
        for (let s = 0; s < 12; s++) {
          const [a0, b0] = p.bracket;
          const u = (Math.sin(s * 12.9898 + a0) * 43758.5453) % 1;
          const a = a0 + Math.abs(u) * 0.3 * (b0 - a0);
          const b = b0 - Math.abs((u * 7.31) % 1) * 0.3 * (b0 - a0);
          if (!(a < b) || Math.sign(p.f(a)) === Math.sign(p.f(b)) || p.f(a) * p.f(b) === 0)
            continue;
          const r = m.fn(p, {
            xtol: 1e-12,
            ftol: 0,
            max_iter: 400,
            kappa1: 0.1,
            kappa2: 2,
            n0: 1,
            bracket: [a, b],
          });
          expect(r.converged).toBe(true);
          for (const st of r.trace) {
            const [lo, hi] = st.info.bracket as number[];
            expect(lo).toBeLessThanOrEqual(hi);
            const [nlo, nhi] = st.info.new_bracket as number[];
            if (nlo !== nhi) expect(Math.sign(p.f(nlo)) * Math.sign(p.f(nhi))).toBeLessThan(0);
            expect(st.x as number).toBeGreaterThanOrEqual(lo);
            expect(st.x as number).toBeLessThanOrEqual(hi);
          }
          const near = Math.min(...p.roots.map((z) => Math.abs(z - (r.x as number))));
          expect(near).toBeLessThan(1e-9);
        }
      }
    });
  }
  it('rejects a bracket without a sign change with Python’s message', () => {
    const { fn } = getMethod('bisection');
    expect(() => fn(getProblem('sqrt2'), { bracket: [2, 3] })).toThrow(
      'f(a) and f(b) must have opposite signs; got f(2.0)=2.0, f(3.0)=7.0',
    );
  });
});
