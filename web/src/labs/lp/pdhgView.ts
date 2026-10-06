/**
 * Shared marks of restarted PDHG in the LP lab: which steps restart, and the restart square
 * drawn on the plane, the polytope and the residual chart (one shape for one meaning).
 */
import type { Step } from '../../core/types';

export const PDHG = 'restarted_pdhg';

/** k of every step that restarted to the epoch average (Step.info.restarted). */
export const restartKs = (trace: readonly Step[]): number[] =>
  trace.filter((s) => s.info.restarted === true).map((s) => s.k);

/** The restart mark: a square in the method's color with a halo, `r` = half side (CSS px). */
export function drawRestartMark(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  color: string,
  halo: string,
  r = 4,
) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = halo;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.rect(px - r, py - r, 2 * r, 2 * r);
  ctx.stroke();
  ctx.fill();
  ctx.restore();
}
