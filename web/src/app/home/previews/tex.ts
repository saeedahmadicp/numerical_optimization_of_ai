/**
 * Numbers inside the `$…$` math of preview titles and captions: a coefficient in a formula, and
 * every power of ten (brand.md §3: powers of ten are KaTeX). A value that stands alone after a
 * relation (`$\mathbf{x}_0 =$ (−3.75, 2.5)`) stays plain text. The plain-text formatters (`sig`,
 * `sci`) write U+2212 and Unicode superscripts; these helpers turn their output into TeX: `-` for
 * the minus and `1.3 \times 10^{-11}` for scientific notation.
 */
import { sci, sig } from '../../../core/format';

/** `−6.2×10⁻⁴` → `-6.2 \times 10^{-4}`; a bare `10⁻⁸` → `10^{-8}`. Other text passes through. */
function toTex(s: string): string {
  return s
    .replace(/×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/g, (_, e: string) => ` \\times 10^{${unsup(e)}}`)
    .replace(/(?<![\d.])10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/g, (_, e: string) => `10^{${unsup(e)}}`)
    .replace(/−/g, '-')
    .replace(/∞/g, '\\infty')
    .replace(/—/g, '\\text{—}');
}

const SUP_DIGITS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const unsup = (e: string) =>
  [...e].map((c) => (c === '⁻' ? '-' : String(SUP_DIGITS.indexOf(c)))).join('');

/** A number to `digits` significant figures, as TeX (scientific outside [10⁻³, 10⁵)). */
export const texNum = (x: number, digits = 4): string => toTex(sig(x, digits));

/** A number in scientific notation, as TeX: `1.3 \times 10^{-11}`. */
export const texSci = (x: number, digits = 3): string => toTex(sci(x, digits));
