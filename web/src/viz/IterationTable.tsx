import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import type { Step } from '../core/types';
import { defaultColumns, type Column } from './columns';
import { SciText } from '../ui/components/Num';
import styles from './IterationTable.module.css';

export interface IterationTableProps {
  steps: readonly Step[];
  /** Current step (highlighted, auto-scrolled). */
  k: number;
  columns?: readonly Column[];
  onSelect?: (k: number) => void;
  rowHeight?: number;
  ariaLabel?: string;
  className?: string;
}

/** A cell value: `m×10ⁿ` in a string is typeset with a real superscript (never mono superscripts). */
const cell = (v: ReactNode) => (typeof v === 'string' ? <SciText text={v} /> : v);

/** Virtualized iteration table: only visible rows are in the DOM; follows the playhead. */
export function IterationTable({
  steps,
  k,
  columns = defaultColumns,
  onSelect,
  rowHeight = 28,
  ariaLabel = 'Iterations',
  className,
}: IterationTableProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(240);
  const userScrolled = useRef(0);
  const header = 30;

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    setHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  // Follow the playhead unless the viewer scrolled in the last 1.5 s.
  useEffect(() => {
    const el = wrap.current;
    if (!el || Date.now() - userScrolled.current < 1500) return;
    const top = k * rowHeight;
    const visTop = el.scrollTop,
      visBottom = el.scrollTop + el.clientHeight - header;
    if (top < visTop || top + rowHeight > visBottom) {
      el.scrollTo({ top: Math.max(0, top - (el.clientHeight - header) / 2 + rowHeight / 2) });
    }
  }, [k, rowHeight]);

  const overscan = 6;
  const first = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const last = Math.min(steps.length - 1, Math.ceil((scrollTop + height) / rowHeight) + overscan);
  const cols = columns.map((c) => c.width ?? 'minmax(80px, 1fr)').join(' ');
  // Minimum table width = sum of the tracks' minimum px sizes + gaps + padding; narrower
  // containers scroll horizontally instead of squeezing cells.
  const minWidth =
    columns.reduce((s, c) => s + Number(/(\d+)px/.exec(c.width ?? '80px')?.[1] ?? 80), 0) +
    10 * (columns.length - 1) +
    24;
  const rows: ReactNode[] = [];
  for (let i = first; i <= last; i++) {
    const s = steps[i];
    rows.push(
      <div
        key={i}
        role="row"
        aria-rowindex={i + 2}
        aria-current={i === k ? 'true' : undefined}
        data-future={i > k ? 'true' : undefined}
        className={styles.row}
        style={{ top: i * rowHeight }}
        onClick={() => onSelect?.(i)}
      >
        {columns.map((c) => (
          <div
            key={c.key}
            role="cell"
            className={`${styles.cell} ${c.align === 'right' ? styles.right : ''}`}
          >
            {cell(c.value(s))}
          </div>
        ))}
      </div>,
    );
  }

  return (
    <div
      ref={wrap}
      className={`${styles.wrap} ${className ?? ''}`}
      style={{ ['--_cols' as string]: cols, ['--_rh' as string]: `${rowHeight}px` }}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
      onWheel={() => (userScrolled.current = Date.now())}
      onTouchMove={() => (userScrolled.current = Date.now())}
      tabIndex={0}
    >
      <div
        role="table"
        aria-label={ariaLabel}
        aria-rowcount={steps.length + 1}
        className={styles.table}
        style={{ minWidth }}
      >
        <div role="row" aria-rowindex={1} className={styles.header}>
          {columns.map((c) => (
            <div
              key={c.key}
              role="columnheader"
              className={`${styles.cell} ${c.align === 'right' ? styles.right : ''}`}
            >
              {c.label}
            </div>
          ))}
        </div>
        <div role="rowgroup" style={{ position: 'relative', height: steps.length * rowHeight }}>
          {rows}
        </div>
      </div>
    </div>
  );
}
