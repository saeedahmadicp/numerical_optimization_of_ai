/**
 * Math on canvases, set in the KaTeX fonts so labels drawn on a chart match the KaTeX formulas
 * beside it: variables italic (KaTeX_Math), vectors bold upright (KaTeX_Main bold), numbers and
 * operators upright (KaTeX_Main), sub/superscripts at 76 % and never below 9.5 px (`scriptSize`:
 * an exponent on an 11-px tick label is 9.5 px, not 8.4 px).
 *
 *   drawMath(ctx, [b('x'), sub('2'), m(' = (0.76, −3.18)')], x, y, { size: 12, color })
 *   drawMath(ctx, pow10Runs(-8), x, y, { size: 11, align: 'right' })      // 10⁻⁸
 *
 * The KaTeX stylesheet (and so its @font-face rules) loads lazily; until the fonts arrive the
 * labels use a serif fallback, and `useCanvas` redraws when they finish loading.
 */
import { preloadKatex } from '../ui/katex';

export type MathStyle = 'main' | 'italic' | 'bold' | 'sans' | 'serif-italic';

export interface MathRun {
  t: string;
  style?: MathStyle;
  script?: 'sub' | 'sup';
}

export const m = (t: string): MathRun => ({ t, style: 'main' });
export const v = (t: string): MathRun => ({ t, style: 'italic' });
export const b = (t: string): MathRun => ({ t, style: 'bold' });
export const sans = (t: string): MathRun => ({ t, style: 'sans' });
export const sub = (t: string, style: MathStyle = 'main'): MathRun => ({ t, style, script: 'sub' });
export const sup = (t: string, style: MathStyle = 'main'): MathRun => ({ t, style, script: 'sup' });

const SERIF = "'Times New Roman', serif";
let requested = false;

export function mathFont(style: MathStyle, size: number): string {
  if (!requested && typeof window !== 'undefined') {
    requested = true;
    void preloadKatex();
  }
  switch (style) {
    case 'italic':
      return `italic 400 ${size}px KaTeX_Math, ${SERIF}`;
    case 'bold':
      return `700 ${size}px KaTeX_Main, ${SERIF}`;
    case 'sans':
      return `500 ${size}px 'Inter Variable', Inter, system-ui, sans-serif`;
    case 'serif-italic':
      return `italic 400 ${size}px 'Newsreader Variable', Newsreader, Georgia, serif`;
    default:
      return `400 ${size}px KaTeX_Main, ${SERIF}`;
  }
}

/** Script scale: at the 12.5-px size of a log-axis tick, 10⁻⁸'s exponent is 9.5 px. */
export const SCRIPT = 0.76;
/** The smallest script on a chart: an exponent or subscript is never set below 9.5 px. */
export const SCRIPT_MIN = 9.5;

/** A sub- or superscript's size for a base size: 76 %, never below SCRIPT_MIN (or the base). */
export const scriptSize = (size: number): number =>
  Math.min(size, Math.max(SCRIPT_MIN, size * SCRIPT));

function runSize(r: MathRun, size: number) {
  return r.script ? scriptSize(size) : size;
}

/** Width of a run list in CSS px. */
export function measureMath(ctx: CanvasRenderingContext2D, runs: readonly MathRun[], size: number) {
  ctx.save();
  let w = 0;
  for (const r of runs) {
    ctx.font = mathFont(r.style ?? 'main', runSize(r, size));
    w += ctx.measureText(r.t).width;
  }
  ctx.restore();
  return w;
}

export interface DrawMathOptions {
  size?: number;
  color?: string;
  align?: 'left' | 'center' | 'right';
  /** Vertical anchor of `y`: the baseline (default) or the middle of the x-height. */
  baseline?: 'alphabetic' | 'middle';
  /** Outline in this color first (3 px), so the label reads over fills. */
  halo?: string;
}

/** Draw the runs; returns the drawn width. */
export function drawMath(
  ctx: CanvasRenderingContext2D,
  runs: readonly MathRun[],
  x: number,
  y: number,
  o: DrawMathOptions = {},
): number {
  const size = o.size ?? 12;
  const width = measureMath(ctx, runs, size);
  let cx = o.align === 'right' ? x - width : o.align === 'center' ? x - width / 2 : x;
  const base = o.baseline === 'middle' ? y + size * 0.3 : y;
  ctx.save();
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  for (const r of runs) {
    const s = runSize(r, size);
    ctx.font = mathFont(r.style ?? 'main', s);
    const dy = r.script === 'sub' ? size * 0.24 : r.script === 'sup' ? -size * 0.42 : 0;
    if (o.halo) {
      ctx.strokeStyle = o.halo;
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.strokeText(r.t, cx, base + dy);
    }
    if (o.color) ctx.fillStyle = o.color;
    ctx.fillText(r.t, cx, base + dy);
    cx += ctx.measureText(r.t).width;
  }
  ctx.restore();
  return width;
}

const MINUS = '−';
/** U+2212 for negatives, as typeset numbers must have. */
export const signed = (n: number | string) => String(n).replace(/^-/, MINUS);

/** 10ⁿ as runs (`1` and `10` for n = 0 and 1). */
export function pow10Runs(exp: number): MathRun[] {
  if (exp === 0) return [m('1')];
  if (exp === 1) return [m('10')];
  return [m('10'), sup(signed(exp))];
}

/** `𝐱ₖ = (a, b)`: a bold vector name with a subscript and a value text. */
export function iterateRuns(name: string, index: string | number, value?: string): MathRun[] {
  const runs: MathRun[] = [b(name), sub(String(index))];
  if (value !== undefined) runs.push(m(` = ${value}`));
  return runs;
}

/** `𝐱⋆ = (1, 1)`. */
export function starRuns(name: string, value?: string): MathRun[] {
  const runs: MathRun[] = [b(name), sup('⋆')];
  if (value !== undefined) runs.push(m(` = ${value}`));
  return runs;
}
