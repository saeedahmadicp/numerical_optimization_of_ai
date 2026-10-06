import type { ReactNode } from 'react';
import { int } from '../core/format';
import styles from './viz.module.css';

export interface SeriesLegendItem {
  /** Color slot (0-based). */
  slot: number;
  /** The method's full name (or `spec.shortName` with `title` set to the full name). */
  label: string;
  /** The full name when `label` is a short name (tooltip). */
  title?: string;
  /** Iterations (or another count), in tabular figures after the name. */
  count?: number;
  /** One more field after the count (a rate guide ρ, an estimate p̂, θ̂). */
  extra?: ReactNode;
}

/**
 * The legend grammar of every convergence chart: a line key in the method color, the full name,
 * the count in tabular figures, then an optional extra field. Charts that draw their own canvas
 * use it so all legends read the same.
 */
export function SeriesLegend({
  items,
  className,
}: {
  items: readonly SeriesLegendItem[];
  className?: string;
}) {
  return (
    <div className={`${styles.legend} ${className ?? ''}`}>
      {items.map((s) => (
        <span key={`${s.label}~${s.slot}`} className={styles.legendItem} title={s.title}>
          <span
            className={styles.legendLine}
            style={{ ['--_c' as string]: `var(--series-${(s.slot % 4) + 1})` }}
          />
          {s.label}
          {s.count !== undefined && <span className={styles.legendCount}>{int(s.count)}</span>}
          {s.extra !== undefined && <span className={styles.legendExtra}>{s.extra}</span>}
        </span>
      ))}
    </div>
  );
}
