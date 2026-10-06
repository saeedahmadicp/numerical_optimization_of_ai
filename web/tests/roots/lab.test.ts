/**
 * The roots lab's display helpers: the estimate of Brent/Chandrupatla/ITP (`info.best`), the
 * correct-digit count of the iteration table, the digits of the filled rule, ITP's σ, the
 * Illinois scale placement, the playback phases and the drag window.
 */
import { describe, expect, it } from 'vitest';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { RootProblem } from '../../src/problems/roots';
import '../../src/labs/roots/setup';
import { estimateOf, isProbe, phase, shownStep, HOLD, FADE } from '../../src/labs/roots/estimate';
import { correctDigits, sigPrefix } from '../../src/labs/roots/columns';
import { cardStep, digitsFor, filledRule } from '../../src/labs/roots/rules';
import { dragWindow } from '../../src/labs/roots/geometry';

const run = (method: string, problem: string, opts: Record<string, unknown> = {}) => {
  const p = getProblem(problem) as RootProblem;
  const { fn } = getMethod(method);
  return fn(p, { bracket: p.bracket, x0: p.x0, ...opts });
};

describe('estimate after a step', () => {
  for (const id of ['brent', 'chandrupatla', 'itp']) {
    it(`${id}: the last estimate is Result.x and beats the last trial point`, () => {
      const r = run(id, 'cubic');
      const last = r.trace[r.trace.length - 1];
      expect(estimateOf(last)).toBe(r.x);
      const root = (getProblem('cubic') as RootProblem).roots[0];
      expect(Math.abs(estimateOf(last) - root)).toBeLessThanOrEqual(
        Math.abs((last.x as number) - root),
      );
    });
  }
  it('brent/cubic: the final trial point is a probe ~1e-10 away, the estimate is exact', () => {
    const r = run('brent', 'cubic');
    const last = r.trace[r.trace.length - 1];
    const root = 2.0945514815423265;
    expect(isProbe(last)).toBe(true);
    expect(Math.abs((last.x as number) - root)).toBeGreaterThan(1e-11);
    expect(Math.abs(estimateOf(last) - root)).toBeLessThan(1e-14);
  });
  it('methods without info.best use x_k', () => {
    const r = run('bisection', 'cubic');
    for (const s of r.trace) {
      expect(estimateOf(s)).toBe(s.x);
      expect(isProbe(s)).toBe(false);
    }
  });
});

describe('correct digits from the error', () => {
  it('counts digits below a round root', () => {
    expect(correctDigits(0.99999999587, 1)).toBe(8);
    expect(correctDigits(1.0000000041, 1)).toBe(8);
  });
  it('does not count a shared prefix as accuracy', () => {
    // 2.0999 shares "2.09" with 2.0945… but its error is 5e-3.
    expect(correctDigits(2.0999, 2.0945514815)).toBe(2);
    expect(correctDigits(2.0946, 2.0945514815)).toBe(4);
  });
  it('is clamped and safe', () => {
    expect(correctDigits(1, 1)).toBe(10);
    expect(correctDigits(5, 1)).toBe(0);
    expect(correctDigits(NaN, 1)).toBe(0);
    expect(correctDigits(1, null)).toBe(0);
    expect(correctDigits(1, 0)).toBe(0);
  });
  it('sigPrefix covers n significant digits of the formatted string', () => {
    expect('0.9999999958'.slice(0, sigPrefix('0.9999999958', 8))).toBe('0.99999999');
    expect('2.094551481'.slice(0, sigPrefix('2.094551481', 4))).toBe('2.094');
    expect('−1.234567890e−05'.slice(0, sigPrefix('−1.234567890e−05', 3))).toBe('−1.23');
    expect(sigPrefix('2.0945', 0)).toBe(0);
    expect(sigPrefix('1.0000000000e−05', 20)).toBe('1.0000000000'.length);
  });
});

/** Numbers in a TeX string (\times 10^{e} folded in). */
const numbers = (tex: string) =>
  [...tex.matchAll(/-?\d+(?:\.\d+)?(?: \\times 10\^\{(-?\d+)\})?/g)].map((m) =>
    m[1] ? Number(m[0].split(' ')[0]) * 10 ** Number(m[1]) : Number(m[0]),
  );

describe('filled rule digits', () => {
  it('digitsFor separates the ends of an interval', () => {
    expect(digitsFor(2, 3)).toBe(4);
    expect(digitsFor(2.09455148144, 2.09455148155)).toBe(13);
    expect(digitsFor(1, 1)).toBe(15);
    expect(digitsFor(0, 0)).toBe(4);
  });
  it('bisection: the operands differ and the midpoint is their mean at every step', () => {
    const r = run('bisection', 'cubic');
    for (const s of r.trace) {
      const tex = filledRule('bisection', cardStep('bisection', r.trace, s.k))!;
      const second = tex.split('\\\\')[1];
      const [a, b] = numbers(second.match(/\\dfrac\{(.*?)\}\{2\}/)![1]);
      expect(a).toBeLessThan(b);
      const [lo, hi] = s.info.bracket as number[];
      expect(Math.abs(a - lo)).toBeLessThan((hi - lo) / 10);
      expect(Math.abs(b - hi)).toBeLessThan((hi - lo) / 10);
      const [, , mid, kept] = tex.split('\\\\');
      const [xm] = numbers(mid);
      expect(xm).toBeGreaterThan(a);
      expect(xm).toBeLessThan(b);
      // The kept half is named: [a_k, x_k] or [x_k, b_k].
      const [nlo, nhi] = s.info.new_bracket as number[];
      const k = s.k;
      expect(kept).toContain(
        nlo === s.x && nhi === s.x
          ? `[x_{${k}},\\ x_{${k}}]`
          : nlo === lo
            ? `[a_{${k}},\\ x_{${k}}]`
            : `[x_{${k}},\\ b_{${k}}]`,
      );
    }
  });
  it('bisection card shows the sign change f(a_k) f(b_k) < 0', () => {
    const r = run('bisection', 'cubic');
    const last = cardStep('bisection', r.trace, r.trace.length - 1)!;
    expect((last.info.fa as number) * (last.info.fb as number)).toBeLessThan(0);
  });
});

describe('ITP shows the sign σ', () => {
  it('uses + or − explicitly, never ±', () => {
    const r = run('itp', 'cubic');
    for (const s of r.trace) {
      const tex = filledRule('itp', s);
      if (!tex) continue;
      expect(tex).not.toMatch(/\\pm|\\mp/);
      const xh = s.info.x_half as number,
        xf = s.info.x_f as number;
      if (s.info.projected === false && (s.info.delta as number) <= Math.abs(xh - xf))
        expect(tex).toContain(xh > xf ? '\\sigma = +1' : '\\sigma = -1');
    }
  });
});

describe('Illinois: the scale is the factor for the next chord', () => {
  it('labels m as "next", not as an input of this step', () => {
    const r = run('illinois', 'x10_minus_1');
    const s = r.trace.find((t) => t.info.scale !== 1 && t.info.step === 'chord')!;
    const tex = filledRule('illinois', s)!;
    expect(tex).toContain('\\text{next }');
    // The scale sits on the third line (after the kept bracket), not on the evaluated line.
    expect(tex.split('\\\\')[1]).not.toContain('next');
  });
});

describe('playback phases', () => {
  it('holds step k, then fades, then travels', () => {
    expect(phase(0)).toEqual({ fade: 0, travel: 0 });
    expect(phase(HOLD - 1e-9)).toEqual({ fade: 0, travel: 0 });
    expect(phase(HOLD + FADE).fade).toBe(1);
    expect(phase(HOLD + FADE).travel).toBeLessThan(1e-12);
    expect(phase(0.999).travel).toBeGreaterThan(0.99);
  });
  it('the card switches step halfway through the cross-fade', () => {
    expect(shownStep(3)).toBe(3);
    expect(shownStep(3 + HOLD)).toBe(3);
    expect(shownStep(3 + HOLD + FADE / 2)).toBe(4);
    expect(shownStep(3.99)).toBe(4);
    expect(shownStep(Infinity)).toBe(Infinity);
  });
});

describe('drag window', () => {
  it('keeps a usable view', () => {
    expect(dragWindow([1, 4], [0, 4], 2)).toEqual([1, 4]);
  });
  it('replaces a view zoomed out after divergence', () => {
    const w = dragWindow([-130, 5], [-3, 3], 1.5);
    expect(w).toEqual([-3, 3]);
  });
  it('replaces a view zoomed in at convergence and holds an off-view handle', () => {
    const w = dragWindow([2.0945, 2.0946], [1, 3], 0.5);
    expect(w[0]).toBeLessThanOrEqual(0.5);
    expect(w[1]).toBe(3);
  });
});
