/** Number formats of the systems lab. */
import { sig } from '../../core/format';

const SUPER: Record<string, string> = {
  '-': '⁻',
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
};

/** α = 2⁻ⁱ as "2⁻ⁱ" (1 as "1"); anything else with 3 significant digits. */
export function powerOfTwo(a: number): string {
  if (a === 1) return '1';
  const e = Math.log2(a);
  if (a > 0 && Number.isInteger(e)) return '2' + [...String(e)].map((c) => SUPER[c]).join('');
  return sig(a, 3);
}
