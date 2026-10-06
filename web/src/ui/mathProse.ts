/**
 * Helpers for prose with inline math (rendered by `MathText`): split `$…$` spans, test for them,
 * flatten them to plain text, and mark the TeX-style scripts of catalog strings as math.
 */
import { texToText } from './texText';

/** Split `text` into alternating prose and TeX parts (odd indices are TeX); `\$` is a dollar. */
export function splitMath(text: string): string[] {
  const parts = text.split(/(?<!\\)\$/);
  // An unpaired `$` is not math: keep the text whole.
  if (parts.length % 2 === 0) return [text];
  return parts;
}

/** True when `text` holds at least one `$…$` span. */
export function hasMath(text: string): boolean {
  return splitMath(text).length > 1;
}

/** A plain-text version of `$…$` prose (for `title` attributes and accessible names). */
export function mathTextToPlain(text: string): string {
  return splitMath(text)
    .map((part, i) => (i % 2 === 0 ? part : texToText(part)))
    .join('');
}

/**
 * Wrap the TeX-style scripts of a plain-text catalog string (`f_{k+1}`, `g_kᵀy`, `h^{2k+2}`,
 * `L_max`, `I_(k−1)`, `ε^(-3/2)`) in `$…$`, so `MathText` sets them as mathematics. A script is a
 * single letter (not part of a word such as `CG_DESCENT`) followed by `_` or `^` and a braced
 * group, a parenthesized group (set as a braced one) or a run of letters and digits; a following
 * `ᵀ` joins it as `^\top`. Text that already has `$…$` spans is converted between them only.
 */
export function scriptsToMath(text: string): string {
  if (!/[_^]/.test(text)) return text;
  const parts = splitMath(text);
  if (parts.length > 1)
    return parts.map((p, i) => (i % 2 === 0 ? scriptsInProse(p) : `$${p}$`)).join('');
  return scriptsInProse(text);
}

function scriptsInProse(text: string): string {
  if (!/[_^]/.test(text)) return text;
  return text.replace(
    /(?<![\p{L}_])([\p{L}])((?:[_^](?:\{[^{}$]*\}|\([^()$]*\)|[A-Za-z0-9]+))+)(ᵀ?)/gu,
    (_, base: string, scripts: string, top: string) => {
      const body = scripts
        .replace(/([_^])\(([^()]*)\)/g, (_m, op: string, g: string) => `${op}{${g}}`)
        .replace(/([_^])([A-Za-z0-9]+)/g, (_m, op: string, s: string) =>
          s.length > 1 && /^[a-z]{2,}$/.test(s) ? `${op}{\\mathrm{${s}}}` : `${op}{${s}}`,
        )
        .replace(/−/g, '-');
      return `$${base}${body}${top ? '^{\\top}' : ''}$`;
    },
  );
}
