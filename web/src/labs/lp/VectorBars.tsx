/**
 * The iterate as numbers when there is no picture (n ≥ 4 variables): one row per variable, one
 * bar per method at the playhead (eased between steps), and the optimum x⋆ⱼ as a tick. Beale's
 * cycling example and the transportation problem live here, beside their tableaux.
 */
import type { Step } from '../../core/types';
import { Formula } from '../../ui/components/Formula';
import { Swatch } from '../../ui/components/MethodChip';
import { segmentAt, lerp } from '../../play/timeline';
import { fmt } from './explain';
import styles from './LPLab.module.css';

export interface BarRun {
  id: string;
  name: string;
  slot: number;
  trace: Step[];
}

export function VectorBars({
  runs,
  t,
  ease,
  optimum,
  n,
}: {
  runs: BarRun[];
  t: number;
  ease: boolean;
  optimum: number[] | null;
  n: number;
}) {
  const at = (r: BarRun): number[] | null => {
    const pts = r.trace.map((s) => s.x as number[] | null);
    if (!pts.length) return null;
    const { i, u } = segmentAt(t, pts.length, ease);
    const p = pts[i],
      q = pts[Math.min(i + 1, pts.length - 1)];
    if (!p) return q ?? null;
    if (!q) return p;
    return p.map((v, j) => lerp(v, q[j], u));
  };
  const values = runs.map(at);
  const all = [
    ...runs.flatMap((r) => r.trace.flatMap((s) => ((s.x as number[] | null) ?? []).map(Math.abs))),
    ...(optimum ?? []).map(Math.abs),
  ].filter(Number.isFinite);
  const max = Math.max(1e-12, ...all);
  return (
    <div
      className={styles.bars}
      role="table"
      aria-label={`Iterate x at the playhead, ${n} variables`}
    >
      <div role="row" className={styles.barsHead}>
        <span role="columnheader">
          <Formula tex="j" />
        </span>
        <span role="columnheader">
          <Formula tex="x_j" /> at the playhead · <Formula tex="x^{\star}_j" /> as a tick
        </span>
      </div>
      {Array.from({ length: n }, (_, j) => (
        <div role="row" key={j} className={styles.barRow}>
          <span role="rowheader" className={styles.barLabel}>
            <Formula tex={`x_{${j + 1}}`} />
          </span>
          <div role="cell" className={styles.barTrack}>
            {runs.map((r, i) => {
              const v = values[i]?.[j];
              return (
                <div key={r.id} className={styles.barLine}>
                  <Swatch slot={r.slot} size={7} />
                  <div className={styles.barArea}>
                    <span
                      className={styles.bar}
                      style={{
                        width: `${v === undefined ? 0 : (Math.max(0, Math.abs(v)) / max) * 100}%`,
                        background: `var(--series-${(r.slot % 4) + 1})`,
                      }}
                    />
                    {optimum && (
                      <span
                        className={styles.star}
                        style={{ left: `${(Math.abs(optimum[j]) / max) * 100}%` }}
                        aria-hidden="true"
                      />
                    )}
                  </div>
                  <span className={styles.barValue}>{v === undefined ? '—' : fmt(v, 4)}</span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
