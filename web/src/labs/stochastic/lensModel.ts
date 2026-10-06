/**
 * The step lens's framing and description (pure, tested): the square box it magnifies and the
 * accessible text that names what it draws.
 */
import { ELLIPSE_R, type StepGeometry } from './geometry';

/** Box (data units) around the step's geometry, square, padded for the labels. */
export function lensBox(g: StepGeometry): { cx: number; cy: number; half: number } {
  const pts: number[][] = [g.base, g.to, g.pushTo, g.evalAt];
  if (g.mean) pts.push(g.mean);
  if (g.gradDir) pts.push(g.gradDir);
  if (g.chain) pts.push(...g.chain);
  if (g.cov && g.mean) {
    const rx = ELLIPSE_R * Math.sqrt(Math.max(0, g.cov[0][0]));
    const ry = ELLIPSE_R * Math.sqrt(Math.max(0, g.cov[1][1]));
    pts.push([g.mean[0] - rx, g.mean[1] - ry], [g.mean[0] + rx, g.mean[1] + ry]);
  }
  const xs = pts.map((p) => p[0]).filter(Number.isFinite);
  const ys = pts.map((p) => p[1]).filter(Number.isFinite);
  const x0 = Math.min(...xs),
    x1 = Math.max(...xs),
    y0 = Math.min(...ys),
    y1 = Math.max(...ys);
  // 0.5 would touch the edges: 0.72 leaves ≈ 30 px of a 196 px lens around the marks for labels.
  const half = Math.max(x1 - x0, y1 - y0, 1e-12) * 0.72;
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, half };
}

/** The lens's accessible description, from what it actually draws. */
export function lensAria(g: StepGeometry, k: number): string {
  const parts = ['the update'];
  if (g.kind === 'momentum' || g.kind === 'nesterov') parts.push('the momentum push βv');
  if (g.kind === 'nesterov') parts.push('the look-ahead point');
  if (g.snapshot) parts.push('the snapshot w̃');
  if (g.mean) parts.push('the full-gradient step −η∇f');
  if (g.gradDir) parts.push('the direction −∇f drawn at the step’s length');
  if (g.cov) parts.push('the 2σ ellipse of the mini-batch step');
  if (g.chain) parts.push('its per-sample parts head to tail');
  const list =
    parts.length > 1 ? `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}` : parts[0];
  return `Step ${k} magnified, with the level sets of f through both iterates: ${list}.`;
}
