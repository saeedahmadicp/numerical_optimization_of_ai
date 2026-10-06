import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { Rng } from '../src/core/rng';

interface Case {
  seed: number;
  random: number[];
  uniform: number[];
  normal: number[];
  integers: number[];
  permutation: number[][];
  choice: string[];
}
const fixture = JSON.parse(
  readFileSync(new URL('./fixtures/rng_python.json', import.meta.url), 'utf8'),
) as { cases: Case[] };

describe('Rng (Mulberry32) matches numopt.core.rng bit for bit', () => {
  it('seed 42 → documented first values', () => {
    const r = new Rng(42);
    expect(r.random()).toBe(0.6011037519201636);
    expect(r.random()).toBe(0.44829055899754167);
    expect(r.random()).toBe(0.8524657934904099);
  });

  for (const c of fixture.cases) {
    it(`seed ${c.seed}: ${c.random.length} uniforms identical to Python`, () => {
      const r = new Rng(c.seed);
      const got = c.random.map(() => r.random());
      expect(got).toEqual(c.random);
    });
    it(`seed ${c.seed}: uniform / integers / permutation / choice identical`, () => {
      let r = new Rng(c.seed);
      expect(c.uniform.map(() => r.uniform(-3, 5))).toEqual(c.uniform);
      r = new Rng(c.seed);
      const ns = [
        ...Array.from({ length: 50 }, (_, i) => i + 1),
        ...Array.from({ length: 50 }, (_, i) => i + 1),
      ];
      expect(ns.map((n) => r.integers(n))).toEqual(c.integers);
      r = new Rng(c.seed);
      expect([1, 2, 5, 10, 31].map((n) => r.permutation(n))).toEqual(c.permutation);
      r = new Rng(c.seed);
      const items = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
      expect(c.choice.map(() => r.choice(items))).toEqual(c.choice);
    });
    it(`seed ${c.seed}: normal matches to libm precision`, () => {
      const r = new Rng(c.seed);
      for (const want of c.normal) {
        const got = r.normal(1.5, 2);
        expect(Math.abs(got - want)).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(want)));
      }
    });
  }
});
