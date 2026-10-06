/**
 * Declarative overlays for one step's geometry (geometry.ts), shared by the landscape (small,
 * unlabelled) and the step lens (magnified). The lens places its labels itself (`stepLabels` +
 * lensLabels.ts) so they never collide; `labels: true` still attaches them to the overlays.
 */
import type { MathRun, Overlay2D } from '../../viz';
import { mathBold, mathMain, mathVar } from '../../viz';
import { ELLIPSE_R, inverse2, type Pt, type StepGeometry } from './geometry';

export interface StepOverlayOptions {
  labels: boolean;
  /**
   * Lens detail: the per-sample chain (sub-pixel on the landscape), the SVRG snapshot, the
   * Nesterov look-ahead and the cross at the expected position.
   */
  chain: boolean;
  /** Arrows shorter than this (data units) are skipped (an arrowhead on nothing). */
  minLength: number;
}

export function stepOverlays(g: StepGeometry, slot: number, o: StepOverlayOptions): Overlay2D[] {
  const out: Overlay2D[] = [];
  const long = (a: readonly number[], c: readonly number[]) =>
    Math.hypot(c[0] - a[0], c[1] - a[1]) > o.minLength;
  const L = <T>(label: T) => (o.labels ? label : undefined);
  if (g.cov && g.mean) {
    const M = inverse2(g.cov);
    if (M)
      out.push({
        kind: 'ellipse',
        center: g.mean,
        matrix: M,
        radius: ELLIPSE_R,
        slot,
        fill: true,
        alpha: 0.9,
        width: 1.25,
        dashed: true,
      });
  }
  if (g.snapshot && (o.labels || o.chain)) {
    out.push({ kind: 'segment', from: g.snapshot, to: g.base, dashed: true, width: 1, alpha: 0.5 });
    out.push({
      kind: 'point',
      at: g.snapshot,
      shape: 'ring',
      radius: 3.5,
      label: L([mathBold('w̃'), mathMain(' snapshot')]),
      labelSide: 'below',
    });
  }
  if ((g.kind === 'momentum' || g.kind === 'nesterov') && long(g.base, g.pushTo))
    out.push({
      kind: 'arrow',
      from: g.base,
      to: g.pushTo,
      slot,
      dashed: true,
      width: 1.25,
      label: L([mathVar('β'), mathBold('v')]),
    });
  if (g.kind === 'nesterov' && (o.labels || o.chain))
    out.push({
      kind: 'point',
      at: g.evalAt,
      shape: 'ring',
      radius: 3,
      slot,
      label: L([mathBold('w̃'), mathMain(' look-ahead')]),
      labelSide: 'above',
    });
  if (g.mean && long(g.pushTo, g.mean)) {
    out.push({ kind: 'arrow', from: g.pushTo, to: g.mean, dashed: true, width: 1.25 });
    // Mark the expected position at the ellipse center (labelled on the side away from the
    // actual step).
    if (o.labels || o.chain) {
      const side = g.to[0] > g.mean[0] ? 'left' : 'right';
      out.push({
        kind: 'point',
        at: g.mean,
        shape: 'cross',
        radius: 3,
        label: L([mathMain('−'), mathVar('η'), mathMain('∇'), mathVar('f')]),
        labelSide: side,
      });
    }
  }
  if (g.gradDir && long(g.base, g.gradDir))
    out.push({
      kind: 'arrow',
      from: g.base,
      to: g.gradDir,
      dashed: true,
      width: 1.25,
      label: L([mathMain('−∇'), mathVar('f')]),
    });
  if (o.chain && g.chain && g.chain.length > 2)
    out.push({ kind: 'polyline', points: g.chain, slot, width: 1, alpha: 0.75 });
  const from = g.kind === 'momentum' || g.kind === 'nesterov' ? g.pushTo : g.base;
  if (long(from, g.to))
    out.push({
      kind: 'arrow',
      from,
      to: g.to,
      slot,
      width: 2,
      label: L(stepRuns(g)),
    });
  out.push({ kind: 'point', at: g.base, shape: 'dot', radius: o.labels || o.chain ? 3 : 2.4 });
  return out;
}

const stepRuns = (g: StepGeometry): MathRun[] =>
  g.kind === 'adaptive'
    ? [mathMain('Δ'), mathBold('w')]
    : [mathMain('−'), mathVar('η'), mathBold(g.kind === 'svrg' || g.kind === 'table' ? 'v' : 'g')];

export type StepLabel =
  { runs: MathRun[]; kind: 'arrow'; from: Pt; to: Pt } | { runs: MathRun[]; kind: 'point'; at: Pt };

/**
 * The lens labels in priority order (the step first), for the marks `stepOverlays` draws with
 * the same `minLength`. Placement is lensLabels.ts's job.
 */
export function stepLabels(g: StepGeometry, minLength: number): StepLabel[] {
  const long = (a: readonly number[], c: readonly number[]) =>
    Math.hypot(c[0] - a[0], c[1] - a[1]) > minLength;
  const out: StepLabel[] = [];
  const momentum = g.kind === 'momentum' || g.kind === 'nesterov';
  const from = momentum ? g.pushTo : g.base;
  if (long(from, g.to)) out.push({ runs: stepRuns(g), kind: 'arrow', from, to: g.to });
  if (g.mean && long(g.pushTo, g.mean))
    out.push({
      runs: [mathMain('−'), mathVar('η'), mathMain('∇'), mathVar('f')],
      kind: 'point',
      at: g.mean,
    });
  if (g.gradDir && long(g.base, g.gradDir))
    out.push({ runs: [mathMain('−∇'), mathVar('f')], kind: 'arrow', from: g.base, to: g.gradDir });
  if (momentum && long(g.base, g.pushTo))
    out.push({ runs: [mathVar('β'), mathBold('v')], kind: 'arrow', from: g.base, to: g.pushTo });
  if (g.kind === 'nesterov') out.push({ runs: [mathBold('w̃')], kind: 'point', at: g.evalAt });
  if (g.snapshot) out.push({ runs: [mathBold('w̃')], kind: 'point', at: g.snapshot });
  return out;
}
