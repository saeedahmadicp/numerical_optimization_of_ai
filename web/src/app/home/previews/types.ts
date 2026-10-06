/**
 * A lab preview: the lab's signature geometry, drawn from real runs of the TS method ports.
 *
 * Every preview module (`./labs/<lab id>.ts`) runs its methods once, when it is first loaded, and
 * returns a `Preview`. `draw` paints the state at progress u ∈ [0, 1] of one play: u = 1 is the
 * poster (the still frame on the home page, and the only frame under reduced motion). Nothing is
 * hand-placed: every point, bracket, tour and curve comes from a `Result.trace`.
 */
import type { ChartColors } from '../../../ui/colors';
import type { CanvasSize } from '../../../viz/useCanvas';

export interface LegendItem {
  /** Textbook name, variant in parentheses. */
  label: string;
  /** Method color slot 0–3. */
  slot: number;
  /** What the run did, e.g. "28 iterations". */
  note: string;
}

export interface Preview {
  /** Lab id (`#/lab/<id>`). */
  lab: string;
  /**
   * What is drawn, e.g. "Himmelblau's function" or "Wallis’ cubic $f(x) = x^3 - 2x - 5$". Math is
   * `$…$` TeX (KaTeX in the hero caption), never Unicode look-alikes such as 𝐱 or x³ (brand.md §3).
   */
  title: string;
  /**
   * Provenance: start, stopping test, counts (brand.md §9). Every symbol, relation, formula and
   * power of ten is `$…$` TeX (`$\mathbf{x}_0 =$ (−3.75, 2.5)`, `$\le 10^{-6}$`, `log $k$`); a
   * value that stands alone (a count, a scalar, a tuple, an interval) stays plain text.
   * `previews.test.ts` checks that every TeX part renders and that the prose has no Unicode math.
   */
  caption: string;
  legend: LegendItem[];
  /** States the outcome for screen readers, in plain Unicode text (no TeX). */
  ariaLabel: string;
  /** Seconds for one play at hero size (cards play a shorter version). */
  duration: number;
  draw(
    ctx: CanvasRenderingContext2D,
    size: CanvasSize,
    c: ChartColors,
    u: number,
    hero: boolean,
  ): void;
}

export type PreviewModule = { default: () => Preview };
