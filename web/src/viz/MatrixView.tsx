import type { ReactNode } from 'react';
import { motion } from 'motion/react';
import { cellText, matrixColumnChars } from './matrixColumns';
import { usePrefersReducedMotion } from '../play/reducedMotion';
import styles from './MatrixView.module.css';
import { baseTransition } from '../ui/motion';

export interface MatrixViewProps {
  matrix: readonly (readonly number[])[];
  /** Row / column labels (strings or KaTeX `<Formula>`s, e.g. basis variables). */
  rowLabels?: readonly ReactNode[];
  colLabels?: readonly ReactNode[];
  /**
   * The matrix before this step: cells whose value changed are marked (a soft wash that fades
   * in 220 ms) in addition to the value cross-fade, so the eye finds what the step touched.
   */
  previous?: readonly (readonly number[])[] | null;
  /** Pivot cell. */
  pivot?: { row: number; col: number } | null;
  /** Rows/columns shaded as the active band (elimination row, entering column, ...). */
  rows?: readonly number[];
  cols?: readonly number[];
  /** Individually highlighted cells `[row, col]` (e.g. the ratio-test winner). */
  cells?: readonly (readonly [number, number])[];
  /** Draw a dashed separator before this column (augmented matrix / RHS of a tableau). */
  separatorBefore?: number;
  digits?: number;
  ariaLabel?: string;
  /**
   * Minimum width of each column in characters of the cell font (`colCh[j]`), so the columns keep
   * their width for a whole run instead of growing when a longer value appears mid-run.
   */
  colCh?: readonly number[];
  /**
   * Every matrix the view will show in this run (all steps of a trace): the column widths are
   * the widest formatted value of each column over all of them. An alternative to `colCh`.
   */
  widthsFrom?: readonly (readonly (readonly number[])[])[];
}

/**
 * A matrix (elimination steps, simplex tableaux) whose changed cells cross-fade their value and
 * flash a soft wash (brand §7: value flash, no slide), and that highlights the pivot row/column.
 * Pass `colCh` or `widthsFrom` so the columns keep one width for the whole run.
 */
export function MatrixView({
  matrix,
  rowLabels,
  colLabels,
  pivot,
  rows = [],
  cols = [],
  cells = [],
  separatorBefore,
  previous,
  digits = 4,
  ariaLabel = 'Matrix',
  colCh,
  widthsFrom,
}: MatrixViewProps) {
  const reduced = usePrefersReducedMotion();
  const n = matrix[0]?.length ?? 0;
  const hasRowLabels = Boolean(rowLabels?.length);
  // Columns sized for the run (colCh / widthsFrom) never grow mid-run; 58 px is the floor.
  const chars = colCh ?? (widthsFrom ? matrixColumnChars(widthsFrom, digits) : null);
  const columns = chars
    ? Array.from(
        { length: n },
        (_, j) => `minmax(max(58px, calc(${chars[j] ?? 0}ch + 16px)), auto)`,
      ).join(' ')
    : `repeat(${n}, auto)`;
  const template = `${hasRowLabels ? 'auto ' : ''}${columns}`;
  const isHi = (i: number, j: number) => cells.some(([r, c]) => r === i && c === j);
  return (
    <div
      role="table"
      aria-label={ariaLabel}
      className={styles.wrap}
      style={{ gridTemplateColumns: template }}
    >
      {colLabels && (
        <div role="row" style={{ display: 'contents' }}>
          {/* The blank corner is a cell, not an unnamed header. */}
          {hasRowLabels && <span role="cell" />}
          {colLabels.map((c, j) => (
            <span key={j} role="columnheader" className={`${styles.label} ${styles.colLabel}`}>
              {c === '' || c === null || c === undefined || c === false ? (
                <span className="visually-hidden">Column {j + 1}</span>
              ) : (
                c
              )}
            </span>
          ))}
        </div>
      )}
      {matrix.map((row, i) => (
        <div role="row" key={i} style={{ display: 'contents' }}>
          {hasRowLabels && (
            <span role="rowheader" className={styles.label} style={{ paddingRight: 6 }}>
              {rowLabels![i]}
            </span>
          )}
          {row.map((v, j) => {
            const text = cellText(v, digits);
            const before = previous?.[i]?.[j];
            const changed = before !== undefined && cellText(before, digits) !== text;
            return (
              <span
                key={j}
                role="cell"
                className={`${styles.cell} ${separatorBefore === j ? styles.sep : ''}`}
                data-zero={text === '0' || undefined}
                data-band={rows.includes(i) || cols.includes(j) || undefined}
                data-pivot={pivot?.row === i && pivot?.col === j ? 'true' : undefined}
                data-hi={isHi(i, j) || undefined}
                data-changed={changed || undefined}
              >
                <motion.span
                  key={text}
                  className={styles.value}
                  initial={reduced ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={baseTransition}
                >
                  {text}
                </motion.span>
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}
