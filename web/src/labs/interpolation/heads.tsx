/**
 * Table headers set in KaTeX with a spoken form in visually hidden text: a header cell needs
 * text of its own (axe: empty-table-header), and "x sub k" reads better than the TeX source.
 */
import { Formula } from '../../ui/components';

/** `\max|p_k - f|` → "max |p k − f|": a plain reading of a short TeX header. */
export function speak(tex: string): string {
  return tex
    .replace(/\\(alpha|beta|ell|omega|lambda)/g, '$1')
    .replace(/\\(max|log)/g, '$1 ')
    .replace(/\\[a-zA-Z]+/g, ' ')
    .replace(/[_^{}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** `header` and `label` for a TableColumn whose header is math. */
export function mathHead(tex: string, spoken = speak(tex)) {
  return {
    header: (
      <>
        <Formula tex={tex} />
        <span className="visually-hidden">{spoken}</span>
      </>
    ),
    label: spoken,
  };
}
