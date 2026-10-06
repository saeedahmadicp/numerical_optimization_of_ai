/**
 * Prose with inline math: `$…$` segments are typeset with KaTeX, the rest is plain text. Method
 * docs (rate, intuition, strengths, weaknesses) use `$…$` so that symbols such as $x_{i+1}$ or
 * $\rho(G_J)$ are set as mathematics, never as raw ASCII LaTeX (portal-direction §4). Text with
 * no (or an unpaired) `$` renders as it is. Helpers: `../mathProse`.
 */
import { Fragment } from 'react';
import { Formula } from './Formula';
import { splitMath } from '../mathProse';

export function MathText({ text }: { text: string }) {
  const parts = splitMath(text);
  if (parts.length === 1) return <>{text}</>;
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 0 ? <Fragment key={i}>{part}</Fragment> : <Formula key={i} tex={part} fallback />,
      )}
    </>
  );
}
