/**
 * Complex arithmetic for the complex-step derivative (`complex_step` in ./methods.ts).
 *
 * Python evaluates the calculus problems at z = x0 + ih with CPython complex arithmetic
 * (`*`, `/`, `**` with a small integer exponent) and NumPy's complex ufuncs (`np.exp`,
 * `np.sin`, `np.sqrt`, which call the C library's cexp/csin/csqrt). The TS problems in
 * src/problems/calculus.ts are real-only, so this module carries a complex-analytic twin of
 * each `f`, written with the same operations in the same order:
 *
 *   - `mul` / `div` / `powi` follow CPython's `_Py_c_prod`, `_Py_c_quot` (Smith's algorithm)
 *     and `c_powi` → `c_powu` (binary powering, starting from 1 + 0i) bit for bit;
 *   - a float operand is promoted to `x + 0i` first, as CPython 3.13 does;
 *   - `exp`, `sin`, `sqrt` use the textbook formulas of glibc's cexp/csin/csqrt for finite
 *     arguments; libm may round their last bit differently from JavaScript's Math functions.
 */

export interface Complex {
  re: number;
  im: number;
}

export const cx = (re: number, im = 0): Complex => ({ re, im });

/** CPython `_Py_c_sum`. */
export const add = (a: Complex, b: Complex): Complex => cx(a.re + b.re, a.im + b.im);

/** CPython `_Py_c_diff`. */
export const sub = (a: Complex, b: Complex): Complex => cx(a.re - b.re, a.im - b.im);

/** CPython `_Py_c_neg`. */
export const neg = (a: Complex): Complex => cx(-a.re, -a.im);

/** CPython `_Py_c_prod`. */
export const mul = (a: Complex, b: Complex): Complex =>
  cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);

/**
 * CPython `_Py_c_quot` (Smith, CACM Algorithm 116). Python raises ZeroDivisionError for a zero
 * divisor; this returns NaN + NaN·i, which is what `complex_step` records for such an error.
 */
export function div(a: Complex, b: Complex): Complex {
  const absRe = b.re < 0 ? -b.re : b.re;
  const absIm = b.im < 0 ? -b.im : b.im;
  if (absRe >= absIm) {
    if (absRe === 0.0) return cx(NaN, NaN);
    const ratio = b.im / b.re;
    const denom = b.re + b.im * ratio;
    return cx((a.re + a.im * ratio) / denom, (a.im - a.re * ratio) / denom);
  }
  if (absIm >= absRe) {
    const ratio = b.re / b.im;
    const denom = b.re * ratio + b.im;
    return cx((a.re * ratio + a.im) / denom, (a.im * ratio - a.re) / denom);
  }
  return cx(NaN, NaN);
}

/** CPython `c_powu`: z^n for an integer n ≥ 1 by binary powering from 1 + 0i. */
function powu(z: Complex, n: number): Complex {
  let r = cx(1.0, 0.0);
  let p = z;
  let mask = 1;
  while (mask > 0 && n >= mask) {
    if (n & mask) r = mul(r, p);
    mask <<= 1;
    p = mul(p, p);
  }
  return r;
}

/** CPython `z ** n` for a small integer n (`c_powi`). */
export function powi(z: Complex, n: number): Complex {
  return n > 0 ? powu(z, n) : div(cx(1.0, 0.0), powu(z, -n));
}

/** e^z = e^a (cos b + i sin b) (glibc cexp, finite arguments). */
export function exp(z: Complex): Complex {
  if (z.im === 0) return cx(Math.exp(z.re), z.im);
  const r = Math.exp(z.re);
  return cx(r * Math.cos(z.im), r * Math.sin(z.im));
}

/** sin z = sin a cosh b + i cos a sinh b (glibc csin, finite arguments). */
export function sin(z: Complex): Complex {
  if (z.im === 0) return cx(Math.sin(z.re), z.im);
  return cx(Math.sin(z.re) * Math.cosh(z.im), Math.cos(z.re) * Math.sinh(z.im));
}

/** Principal square root (glibc csqrt, finite arguments). */
export function sqrt(z: Complex): Complex {
  const { re: a, im: b } = z;
  if (b === 0) {
    return a >= 0
      ? cx(Math.sqrt(a), b)
      : cx(0, b < 0 || Object.is(b, -0) ? -Math.sqrt(-a) : Math.sqrt(-a));
  }
  const t = Math.sqrt((Math.abs(a) + Math.hypot(a, b)) / 2);
  return a >= 0 ? cx(t, b / (2 * t)) : cx(Math.abs(b) / (2 * t), b < 0 ? -t : t);
}

/** `float * z` with the float promoted to x + 0i (CPython 3.13). */
export const scale = (s: number, z: Complex): Complex => mul(cx(s, 0.0), z);
/** `float + z`. */
export const addReal = (s: number, z: Complex): Complex => add(cx(s, 0.0), z);
/** `z - float`. */
export const subReal = (z: Complex, s: number): Complex => sub(z, cx(s, 0.0));

/**
 * Complex-analytic twins of the calculus problems' `f`, keyed by problem id, written exactly as
 * `src/numopt/problems/calculus.py` writes them.
 */
export const COMPLEX_F: Readonly<Record<string, (z: Complex) => Complex>> = {
  // x**3 - 2.0 * x + 1.0
  poly3: (z) => addReal(1.0, sub(powi(z, 3), scale(2.0, z))),
  // np.exp(x)
  exp_0_1: (z) => exp(z),
  // np.sin(x)
  sin_0_pi: (z) => sin(z),
  // 1.0 / (1.0 + 25.0 * x**2)
  runge: (z) => div(cx(1.0, 0.0), addReal(1.0, scale(25.0, powi(z, 2)))),
  // np.sqrt(x)
  sqrt_0_1: (z) => sqrt(z),
  // np.exp(-(x**2))
  gaussian: (z) => exp(neg(powi(z, 2))),
  // np.sin(10.0 * x)
  oscillatory: (z) => sin(scale(10.0, z)),
  // 1.0 / (1.0 + x**2)
  arctan_deriv: (z) => div(cx(1.0, 0.0), addReal(1.0, powi(z, 2))),
  // np.where(np.real(d) >= 0, d, -d) with d = x - 0.3: |d| continued analytically from each side
  abs_kink: (z) => {
    const d = subReal(z, 0.3);
    return d.re >= 0 ? d : neg(d);
  },
};
