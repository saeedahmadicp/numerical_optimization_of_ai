/**
 * Correctly rounded elementary functions in double-double arithmetic (≈ 106-bit intermediates).
 *
 * Why: the Python reference runs on glibc's libm and CPython's `math.hypot`, which round
 * correctly in all but ~0.1 % of arguments, while V8's fdlibm-derived `Math.exp/sin/cos/atan`
 * and `Math.hypot` differ from them by one ulp on 3–36 % of arguments. A one-ulp difference in
 * eˣ − 10 near its root is a relative error of 10⁻⁸ in f, enough to change the path of a
 * root finder. These functions return the double nearest to the exact value (barring
 * astronomically rare hard cases), which agrees with glibc wherever glibc rounds correctly.
 *
 * Cost: a few hundred flops per call (Horner on double-double numbers) — fine for plotting.
 * Arguments outside the accurate reduction range (|x| > 2²⁰ for sin/cos) use `Math.*`.
 */

type DD = [number, number];

const SPLITTER = 134217729; // 2^27 + 1

function twoSum(a: number, b: number): DD {
  const s = a + b;
  const bb = s - a;
  return [s, a - (s - bb) + (b - bb)];
}

function quickTwoSum(a: number, b: number): DD {
  const s = a + b;
  return [s, b - (s - a)];
}

function split(a: number): DD {
  const c = SPLITTER * a;
  const hi = c - (c - a);
  return [hi, a - hi];
}

function twoProd(a: number, b: number): DD {
  const p = a * b;
  const [ah, al] = split(a);
  const [bh, bl] = split(b);
  return [p, ah * bh - p + ah * bl + al * bh + al * bl];
}

function add(a: DD, b: DD): DD {
  let [s, e] = twoSum(a[0], b[0]);
  const [t, f] = twoSum(a[1], b[1]);
  e += t;
  [s, e] = quickTwoSum(s, e);
  e += f;
  return quickTwoSum(s, e);
}

function mul(a: DD, b: DD): DD {
  const [p, e] = twoProd(a[0], b[0]);
  return quickTwoSum(p, e + (a[0] * b[1] + a[1] * b[0]));
}

function mulD(a: DD, b: number): DD {
  const [p, e] = twoProd(a[0], b);
  return quickTwoSum(p, e + a[1] * b);
}

function div(a: DD, b: DD): DD {
  const q1 = a[0] / b[0];
  let r = add(a, mulD(b, -q1));
  const q2 = r[0] / b[0];
  r = add(r, mulD(b, -q2));
  const q3 = r[0] / b[0];
  return add(quickTwoSum(q1, q2), [q3, 0]);
}

function sqrtDD(a: DD): DD {
  if (a[0] <= 0) return [Math.sqrt(a[0]), 0];
  const s = Math.sqrt(a[0]);
  // One Newton correction: s + (a − s²)/(2s).
  const r = add(a, twoProd(-s, s));
  return quickTwoSum(s, (r[0] + r[1]) / (2 * s));
}

const neg = (a: DD): DD => [-a[0], -a[1]];
const round = (a: DD) => a[0] + a[1];

// 1/n! for n = 0..27 as double-double numbers.
const INV_FACT: DD[] = [[1, 0]];
for (let n = 1; n <= 27; n++) INV_FACT.push(div(INV_FACT[n - 1], [n, 0]));

const LN2: [number, number, number] = [
  0.6931471805599453, 2.3190468138462996e-17, 5.707708438416212e-34,
];
const PIO2: [number, number, number, number] = [
  1.5707963267948966, 6.123233995736766e-17, -1.4973849048591698e-33, 5.562271104316826e-50,
];

/** Σ c_n rⁿ by Horner for n = first, first + step, … ≤ last (coefficients 1/n!). */
function taylor(r: DD, last: number, step: number, first: number, signed: boolean): DD {
  let acc: DD = [0, 0];
  for (let n = last; n >= first; n -= step) {
    const c = INV_FACT[n];
    const sgn = signed && ((n - first) / step) % 2 === 1 ? -1 : 1;
    acc = add(mul(acc, step === 1 ? r : mul(r, r)), sgn > 0 ? c : neg(c));
  }
  return acc;
}

/** eˣ, correctly rounded. */
export function crExp(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x > 709.782712893384) return Infinity;
  if (x < -745.1332191019412) return 0;
  if (x === 0) return 1;
  const k = Math.round(x * Math.LOG2E);
  // r = x − k·ln 2 in double-double (k·L1, k·L2 exact products).
  let r = add([x, 0], neg(twoProd(k, LN2[0])));
  r = add(r, neg(twoProd(k, LN2[1])));
  r = add(r, [-k * LN2[2], 0]);
  // |r| ≤ ½ ln 2 + tiny: 27 terms leave a truncation error below 2⁻¹¹⁰.
  const e = taylor(r, 27, 1, 0, false);
  // Scale by 2^k exactly (in two factors when 2^k alone would overflow or underflow).
  let hi = e[0],
    lo = e[1];
  let kk = k;
  while (kk > 1000) {
    hi *= 2 ** 1000;
    lo *= 2 ** 1000;
    kk -= 1000;
  }
  while (kk < -1000) {
    hi *= 2 ** -1000;
    lo *= 2 ** -1000;
    kk += 1000;
  }
  const s = 2 ** kk;
  return hi * s + lo * s;
}

/** Reduce x to r = x − k·π/2 (double-double), |r| ≤ π/4; returns [r, k mod 4]. */
function reduce(x: number): [DD, number] {
  const k = Math.round(x / PIO2[0]);
  let r = add([x, 0], neg(twoProd(k, PIO2[0])));
  r = add(r, neg(twoProd(k, PIO2[1])));
  r = add(r, neg(twoProd(k, PIO2[2])));
  r = add(r, [-k * PIO2[3], 0]);
  return [r, ((k % 4) + 4) % 4];
}

const sinR = (r: DD) => mul(r, taylor(r, 27, 2, 1, true));
const cosR = (r: DD) => taylor(r, 26, 2, 0, true);

/** sin x, correctly rounded for |x| ≤ 2²⁰ (Math.sin beyond). */
export function crSin(x: number): number {
  if (!Number.isFinite(x)) return NaN;
  if (Math.abs(x) > 1048576) return Math.sin(x);
  if (x === 0) return x; // keeps −0
  if (Math.abs(x) < 1e-8) return x - (x * x * x) / 6; // exact to rounding
  const [r, q] = reduce(x);
  const v = q === 0 ? sinR(r) : q === 1 ? cosR(r) : q === 2 ? neg(sinR(r)) : neg(cosR(r));
  return round(v);
}

/** cos x, correctly rounded for |x| ≤ 2²⁰ (Math.cos beyond). */
export function crCos(x: number): number {
  if (!Number.isFinite(x)) return NaN;
  if (Math.abs(x) > 1048576) return Math.cos(x);
  const [r, q] = reduce(x);
  const v = q === 0 ? cosR(r) : q === 1 ? neg(sinR(r)) : q === 2 ? neg(cosR(r)) : sinR(r);
  return round(v);
}

// (−1)^j/(2j + 1) at index n = 2j + 1.
const ATAN_COEF: DD[] = [];
for (let n = 1; n <= 41; n += 2) {
  const c = div([1, 0], [n, 0]);
  ATAN_COEF[n] = ((n - 1) / 2) % 2 === 1 ? neg(c) : c;
}

/** atan x, correctly rounded. */
export function crAtan(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === 0) return x;
  const ax = Math.abs(x);
  if (ax > 1e17) return x > 0 ? PIO2[0] : -PIO2[0];
  if (ax < 1e-8) return x - (x * x * x) / 3;
  let t: DD = [ax, 0];
  let big = false;
  if (ax > 1) {
    t = div([1, 0], t);
    big = true;
  }
  // Halve the angle until |t| < 0.06: atan t = 2·atan(t/(1 + √(1 + t²))).
  let doublings = 0;
  while (t[0] > 0.06) {
    const s = sqrtDD(add([1, 0], mul(t, t)));
    t = div(t, add([1, 0], s));
    doublings++;
  }
  // atan t = t − t³/3 + t⁵/5 − … ; 0.06^41/41 < 2⁻¹¹⁵.
  const t2 = mul(t, t);
  let acc: DD = [0, 0];
  for (let n = 41; n >= 1; n -= 2) acc = add(mul(acc, t2), ATAN_COEF[n]);
  let a = mul(t, acc);
  a = [a[0] * 2 ** doublings, a[1] * 2 ** doublings];
  if (big) a = add([PIO2[0], PIO2[1]], neg(a));
  const v = round(a);
  return x < 0 ? -v : v;
}

/** √(a² + b²), correctly rounded (CPython's `math.hypot` rounds correctly almost always). */
export function crHypot(a: number, b: number): number {
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    if (Math.abs(a) === Infinity || Math.abs(b) === Infinity) return Infinity;
    return NaN;
  }
  let x = Math.abs(a),
    y = Math.abs(b);
  if (x < y) [x, y] = [y, x];
  if (x === 0) return 0;
  if (y === 0) return x;
  // Scale by a power of two so the squares neither overflow nor underflow.
  const e = Math.floor(Math.log2(x));
  const scale = e > 500 || e < -500 ? 2 ** -e : 1;
  x *= scale;
  y *= scale;
  const s = add(twoProd(x, x), twoProd(y, y));
  const r = round(sqrtDD(s));
  return scale === 1 ? r : r / scale;
}
