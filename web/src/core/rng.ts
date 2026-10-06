/**
 * Mulberry32 — bit-identical to `numopt.core.rng.Rng` (Python).
 *
 * Every stochastic method draws ONLY from this generator, in the same documented order as the
 * Python implementation, so a seed gives the same stream in both languages. Not cryptographic.
 */
export class Rng {
  private state: number;

  constructor(seed = 0) {
    // Python: int(seed) & 0xFFFFFFFF. `>>> 0` gives the same unsigned 32-bit value for integers
    // in the safe range, including negative seeds (two's complement).
    this.state = Math.trunc(seed) >>> 0;
  }

  /** Uniform in [0, 1) with 32-bit resolution. */
  random(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(low = 0, high = 1): number {
    return low + (high - low) * this.random();
  }

  /** Box–Muller, one variate per call (the second variate is discarded, as in Python). */
  normal(mean = 0, std = 1): number {
    const u1 = 1 - this.random(); // (0, 1]
    const u2 = this.random();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /** Uniform integer in [0, n). */
  integers(n: number): number {
    return Math.min(Math.trunc(this.random() * n), n - 1);
  }

  /** Fisher–Yates shuffle of 0..n-1. */
  permutation(n: number): number[] {
    const out = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = this.integers(i + 1);
      const tmp = out[i];
      out[i] = out[j];
      out[j] = tmp;
    }
    return out;
  }

  choice<T>(items: readonly T[]): T {
    return items[this.integers(items.length)];
  }
}
