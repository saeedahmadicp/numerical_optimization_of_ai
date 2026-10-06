/**
 * Camera of the 1-D stage: the whole-interval view and the follow view that zooms with the
 * focused method's region of interest (eased in log-width between steps). Pure functions.
 */
import { easeInOut } from '../../play/timeline';
import type { StepView } from './geometry';

const EPS = Number.EPSILON;
/** Follow view = region of interest × PAD (the bracket fills 1/PAD of the width). */
export const PAD = 1.6;

/** The whole-interval view: the plotting domain, widened to hold the inputs. */
export function fullView(
  domain: [number, number],
  bracket: [number, number] | null,
  x0: number | null,
): [number, number] {
  let lo = domain[0],
    hi = domain[1];
  for (const v of [...(bracket ?? []), ...(x0 === null ? [] : [x0])]) {
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (lo < domain[0] || hi > domain[1]) {
    const pad = (hi - lo) * 0.04;
    return [lo - (lo < domain[0] ? pad : 0), hi + (hi > domain[1] ? pad : 0)];
  }
  return [lo, hi];
}

const minWidth = (c: number) => 64 * EPS * Math.max(Math.abs(c), 1e-290);

/** The follow-camera window at local time t of a run (eased in log-width between steps). */
export function followView(
  views: readonly StepView[],
  t: number,
  ease: boolean,
  full: [number, number],
): [number, number] {
  if (views.length === 0) return full;
  const fullW = full[1] - full[0];
  const lt = Math.max(0, Math.min(t, views.length - 1));
  const i = Math.floor(lt);
  const j = Math.min(i + 1, views.length - 1);
  const u = ease ? easeInOut(lt - i) : 0;
  const widthOf = (idx: number): number => {
    const [lo, hi] = views[idx].roi;
    const c = (lo + hi) / 2;
    if (hi - lo > minWidth(c)) return hi - lo;
    // A point (Newton's start): borrow the neighbor's width, else a tenth of the view.
    for (const n of [idx + 1, idx - 1]) {
      const r = views[n]?.roi;
      if (r && r[1] - r[0] > minWidth(c)) return r[1] - r[0];
    }
    return fullW * 0.1;
  };
  const c0 = (views[i].roi[0] + views[i].roi[1]) / 2;
  const c1 = (views[j].roi[0] + views[j].roi[1]) / 2;
  const w0 = widthOf(i),
    w1 = widthOf(j);
  const c = c0 + (c1 - c0) * u;
  const w = Math.min(fullW, Math.exp(Math.log(w0) + (Math.log(w1) - Math.log(w0)) * u) * PAD);
  let lo = c - w / 2,
    hi = c + w / 2;
  if (lo < full[0]) [lo, hi] = [full[0], full[0] + w];
  if (hi > full[1]) [lo, hi] = [full[1] - w, full[1]];
  return [lo, hi];
}
