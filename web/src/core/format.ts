/** Number formatting for readouts, tables and axes. Uses U+2212 minus and unicode superscripts. */

const SUP: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '-': '⁻',
  '+': '',
};
export const MINUS = '−';

export function superscript(n: number | string): string {
  return String(n)
    .split('')
    .map((c) => SUP[c] ?? c)
    .join('');
}

function special(x: number | null | undefined): string | null {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  if (x === Infinity) return '∞';
  if (x === -Infinity) return `${MINUS}∞`;
  return null;
}

const fixMinus = (s: string) => s.replace(/^-/, MINUS);

/** Scientific notation with unicode superscripts: `1.23×10⁻⁸`. */
export function sci(x: number | null | undefined, digits = 3): string {
  const s = special(x);
  if (s) return s;
  const v = x as number;
  if (v === 0) return '0';
  const [mant, exp] = v.toExponential(Math.max(0, digits - 1)).split('e');
  const e = Number(exp);
  const m = mant.replace(/\.?0+$/, '');
  if (e === 0) return fixMinus(m);
  return `${fixMinus(m)}×10${superscript(e)}`;
}

/** Significant figures, switching to scientific outside [1e-3, 1e5). Trailing zeros trimmed. */
export function sig(x: number | null | undefined, digits = 4): string {
  const s = special(x);
  if (s) return s;
  const v = x as number;
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return sci(v, digits);
  const str = v.toPrecision(digits);
  return fixMinus(str.includes('.') ? str.replace(/\.?0+$/, '') : str);
}

/** Fixed significant figures WITHOUT trimming (for aligned table columns). */
export function sigFixed(x: number | null | undefined, digits = 6): string {
  const s = special(x);
  if (s) return s;
  const v = x as number;
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-4 || a >= 1e6)) {
    const [mant, exp] = v.toExponential(digits - 1).split('e');
    return `${fixMinus(mant)}e${Number(exp) < 0 ? MINUS : '+'}${String(Math.abs(Number(exp))).padStart(2, '0')}`;
  }
  return fixMinus(v.toPrecision(digits));
}

/** A vector as `(1.23, −4.5)`. */
export function vec(x: readonly number[] | number | null | undefined, digits = 4): string {
  if (x === null || x === undefined) return '—';
  if (typeof x === 'number') return sig(x, digits);
  return `(${x.map((v) => sig(v, digits)).join(', ')})`;
}

/**
 * A vector with fixed significant figures for aligned table columns: `( 0.46170,  0.21047)`.
 * Non-negative entries get a leading space so signs line up in a monospaced face.
 */
export function vecFixed(x: readonly number[] | number | null | undefined, digits = 5): string {
  if (x === null || x === undefined) return '—';
  const one = (v: number) => {
    const t = sigFixed(v, digits);
    return v >= 0 || Number.isNaN(v) ? ` ${t}` : t;
  };
  if (typeof x === 'number') return one(x);
  return `(${x.map(one).join(',')})`;
}

/** Axis tick label: as short as possible given the tick step. */
export function tick(x: number, step: number): string {
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return sci(x, 2);
  const decimals = Math.max(0, Math.min(8, -Math.floor(Math.log10(step) + 1e-9)));
  return fixMinus(x.toFixed(decimals));
}

/** Log-axis tick label: `10⁻⁴`, `1`, `10`. */
export function powTen(exp: number): string {
  if (exp === 0) return '1';
  if (exp === 1) return '10';
  return `10${superscript(exp)}`;
}

export function int(n: number): string {
  return n.toLocaleString('en-US');
}

const FROM_SUP: Record<string, string> = {
  '⁰': '0',
  '¹': '1',
  '²': '2',
  '³': '3',
  '⁴': '4',
  '⁵': '5',
  '⁶': '6',
  '⁷': '7',
  '⁸': '8',
  '⁹': '9',
  '⁻': MINUS,
};
/** `6.17×10⁻¹⁰`, `−6×10⁻⁵`, `1×10⁸` or a bare power `10⁻⁸` inside any text. */
const SCI_TOKEN = /(?<![\d.×])(?:([−-]?\d+(?:\.\d+)?)×)?10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/g;

/** True when `text` holds a power of ten with a superscript exponent (`1.3×10⁻⁸`, `10⁻⁸`). */
export const hasSci = (text: string) => /10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]/.test(text);

/**
 * Split text at its scientific-notation numbers: plain runs stay strings, each `m×10ⁿ` becomes
 * `{mantissa, exponent}` with the exponent in plain digits (`−10`), ready for a real superscript.
 */
export function splitSci(text: string): (string | { mantissa: string; exponent: string })[] {
  const out: (string | { mantissa: string; exponent: string })[] = [];
  // `mantissa` is '' for a bare power of ten (10⁻⁸).
  let last = 0;
  for (const m of text.matchAll(SCI_TOKEN)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    out.push({
      mantissa: m[1] ? fixMinus(m[1]) : '',
      exponent: m[2]
        .split('')
        .map((c) => FROM_SUP[c] ?? c)
        .join(''),
    });
    last = at + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/**
 * A parameter value as typeset text: counts grouped (`100,000`), tolerances as powers of ten
 * (`10⁻⁸`, `5×10⁻⁴`), other floats to 4 significant digits (`0.9`, `1.5`). Never `1e-08`.
 */
export function paramValue(v: number, kind: 'int' | 'float' | string = 'float'): string {
  const s = special(v);
  if (s) return s;
  if (kind === 'int') return fixMinus(int(v));
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-3 || a >= 1e5)) {
    const t = sci(v, 3);
    return t.replace(/^(−?)1×10/, '$110');
  }
  return sig(v, 4);
}
