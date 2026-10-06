/**
 * Prose with inline math: `$…$` segments are typeset with KaTeX, the rest is text. The lab's
 * method docs (intuition, strengths, failures) use it so that symbols such as $x_{i+1}$ or
 * $\rho(G_J)$ are set as mathematics, never as raw ASCII LaTeX (portal-direction §4). It is the
 * shared `MathText` (the generic MethodCard uses it too).
 */
import { MathText } from '../../ui/components/MathText';

export function MathProse({ text }: { text: string }) {
  return <MathText text={text} />;
}
