/**
 * Prose with inline math: `$…$` segments are typeset with KaTeX, the rest is Inter text, so a
 * note reads "the shift $\tau = 0.5$ made …" with the symbols set exactly as in the formulas.
 */
import { Fragment } from 'react';
import { Formula } from '../../ui/components/Formula';

/** Split on unescaped `$`: even indices are text, odd indices are TeX. */
function splitMath(text: string): string[] {
  return text.split(/(?<!\\)\$/);
}

export function MathText({ text }: { text: string }) {
  return (
    <>
      {splitMath(text).map((part, i) =>
        i % 2 === 1 ? <Formula key={i} tex={part} fallback /> : <Fragment key={i}>{part}</Fragment>,
      )}
    </>
  );
}
