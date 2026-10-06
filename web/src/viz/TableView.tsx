import { useEffect, useRef, type ReactNode } from 'react';
import { Formula } from '../ui/components/Formula';
import styles from './TableView.module.css';

export interface TableColumn<R> {
  key: string;
  /** Header as KaTeX (`'\\mathbf{x}_k'`, `'\\|\\nabla f\\|'`); takes precedence over `header`. */
  tex?: string;
  /** Header as plain text or a node (words: "Nodes", "Error"). */
  header?: ReactNode;
  /** Accessible header text when the header is math (default: the TeX source). */
  label?: string;
  value: (row: R, index: number) => ReactNode;
  /** Numbers right-aligned in a tabular mono face (default 'right'). */
  align?: 'left' | 'right' | 'center';
  /** Set the cells in JetBrains Mono (default true for right-aligned columns). */
  mono?: boolean;
  /** CSS width of the column (e.g. `4rem`). */
  width?: string;
}

export interface TableViewProps<R> {
  columns: readonly TableColumn<R>[];
  rows: readonly R[];
  /** Highlighted row (soft iris fill + a 2 px rule on the left), scrolled into view. */
  highlight?: number | null;
  /** Rows after this index are drawn faint (the future of a replay). */
  futureAfter?: number;
  onSelect?: (index: number) => void;
  /** Visible caption above the table (also its accessible name). */
  caption?: ReactNode;
  ariaLabel?: string;
  /** Scroll inside a box of this height (CSS); the header stays put. */
  maxHeight?: number | string;
  rowKey?: (row: R, index: number) => string | number;
  className?: string;
  /** Text when there are no rows. */
  empty?: ReactNode;
}

/**
 * A small semantic table with KaTeX headers (`<table>`, not virtualized: use IterationTable for
 * thousands of rows). Numbers are right-aligned in tabular figures; the current row follows the
 * playhead.
 */
export function TableView<R>({
  columns,
  rows,
  highlight = null,
  futureAfter,
  onSelect,
  caption,
  ariaLabel,
  maxHeight,
  rowKey,
  className,
  empty = 'No rows.',
}: TableViewProps<R>) {
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (highlight === null || !scroller.current) return;
    const row = scroller.current.querySelector<HTMLElement>(`[data-row="${highlight}"]`);
    if (!row) return;
    const box = scroller.current;
    const head = box.querySelector('thead')?.getBoundingClientRect().height ?? 0;
    const top = row.offsetTop - head,
      bottom = row.offsetTop + row.offsetHeight;
    if (top < box.scrollTop || bottom > box.scrollTop + box.clientHeight)
      box.scrollTo({ top: Math.max(0, top - box.clientHeight / 2) });
  }, [highlight]);

  return (
    <div
      ref={scroller}
      className={`${styles.wrap} ${className ?? ''}`}
      style={maxHeight !== undefined ? { maxHeight } : undefined}
      tabIndex={maxHeight !== undefined ? 0 : undefined}
      role={maxHeight !== undefined ? 'region' : undefined}
      aria-label={maxHeight !== undefined ? (ariaLabel ?? 'Table') : undefined}
    >
      <table className={styles.table} aria-label={caption ? undefined : ariaLabel}>
        {caption && <caption className={styles.caption}>{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={styles.th}
                data-align={c.align ?? 'right'}
                style={c.width ? { width: c.width } : undefined}
                aria-label={c.tex ? (c.label ?? c.tex) : c.label}
              >
                {c.tex ? <Formula tex={c.tex} /> : c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td className={styles.empty} colSpan={columns.length}>
                {empty}
              </td>
            </tr>
          )}
          {rows.map((r, i) => (
            <tr
              key={rowKey ? rowKey(r, i) : i}
              data-row={i}
              className={styles.tr}
              aria-current={i === highlight ? 'true' : undefined}
              data-future={futureAfter !== undefined && i > futureAfter ? true : undefined}
              data-clickable={onSelect ? true : undefined}
              onClick={onSelect ? () => onSelect(i) : undefined}
            >
              {columns.map((c) => {
                const align = c.align ?? 'right';
                return (
                  <td
                    key={c.key}
                    className={styles.td}
                    data-align={align}
                    data-mono={(c.mono ?? align === 'right') || undefined}
                  >
                    {c.value(r, i)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
