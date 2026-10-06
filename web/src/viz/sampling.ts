/**
 * Adaptive sampling: start uniform, then bisect any interval whose midpoint deviates from the
 * chord by more than `tol` pixels (up to `maxDepth`), so kinks and steep parts stay smooth.
 */
export function adaptiveSample(
  f: (x: number) => number,
  a: number,
  b: number,
  toPx: (x: number, y: number) => [number, number],
  tol = 0.35,
  n0 = 96,
  maxDepth = 9,
): [number, number][] {
  const out: [number, number][] = [];
  const ev = (x: number): [number, number] => [x, f(x)];
  const rec = (p: [number, number], q: [number, number], depth: number) => {
    const m = ev((p[0] + q[0]) / 2);
    if (depth < maxDepth) {
      const P = toPx(p[0], p[1]),
        Q = toPx(q[0], q[1]),
        Mp = toPx(m[0], m[1]);
      const finite = [P, Q, Mp].every(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      const dev = finite
        ? Math.abs((Q[0] - P[0]) * (P[1] - Mp[1]) - (P[0] - Mp[0]) * (Q[1] - P[1])) /
          Math.max(1e-9, Math.hypot(Q[0] - P[0], Q[1] - P[1]))
        : Infinity;
      if (dev > tol || !finite) {
        rec(p, m, depth + 1);
        rec(m, q, depth + 1);
        return;
      }
    }
    out.push(q);
  };
  let prev = ev(a);
  out.push(prev);
  for (let i = 1; i <= n0; i++) {
    const next = ev(a + ((b - a) * i) / n0);
    rec(prev, next, 0);
    prev = next;
  }
  return out;
}
