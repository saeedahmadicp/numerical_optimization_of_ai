/**
 * The Romberg table R(k, j) filling in row by row with the playhead. Column 0 is the trapezoid
 * rule on 2^k panels, column 1 Simpson, column 2 Boole; each entry is
 * R(k, j) = R(k, j−1) + (R(k, j−1) − R(k−1, j−1))/(4^j − 1). The digits that agree with the exact
 * integral are set in ink, the rest in a quiet tone.
 *
 * The full triangle is wider than the details column, so the estimate R(k, k) — the diagonal —
 * also has its own pinned column next to k: reading it downwards shows the digits being gained.
 * The triangle scrolls sideways under it and follows the playhead (the current row's diagonal
 * cell is kept in view); a fade at the right edge says that more columns are there. The caption
 * stays above the scroller.
 */
import { useEffect, useRef, useState } from 'react';
import type { Step } from '../../core/types';
import { Formula } from '../../ui/components';
import { agreeingDigits } from './quadFormat';
import styles from './IntegrationLab.module.css';

const COLUMN_NAMES = ['trapezoid', 'Simpson', 'Boole'];
const DIGITS = 10;

function Digits({ v, exact }: { v: number; exact: number | null }) {
  const d = agreeingDigits(v, exact, DIGITS);
  return (
    <>
      <span className={styles.good}>{d.good}</span>
      <span className={styles.rest}>{d.rest}</span>
    </>
  );
}

export function RombergTable({
  trace,
  k,
  exact,
  onSelect,
}: {
  trace: readonly Step[];
  k: number;
  exact: number | null;
  onSelect?: (k: number) => void;
}) {
  const K = trace.length - 1;
  const cols = Math.min(K, 12) + 1;
  const box = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: false });

  const measure = () => {
    const el = box.current;
    if (!el) return;
    const left = el.scrollLeft > 1;
    // The trailing padding (one column, so the last columns can start at the pinned edge too)
    // is not "more".
    const pad = parseFloat(getComputedStyle(el).paddingRight) || 0;
    const right = el.scrollLeft + el.clientWidth < el.scrollWidth - pad - 1;
    setEdges((e) => (e.left === left && e.right === right ? e : { left, right }));
  };

  // Keep the current diagonal cell R(k, k) in view at the right, with as many of the columns
  // before it as fit, and start the visible part on a whole column next to the pinned ones.
  // Runs when the row changes and when the table resizes (web fonts and KaTeX widen columns).
  const follow = () => {
    const el = box.current;
    const cell = el?.querySelector<HTMLElement>('td[data-current]');
    const pinned = el?.querySelector<HTMLElement>('thead th[data-pin="diag"]');
    if (!el) return;
    if (cell) {
      // The pinned edge on screen (a sticky cell's offsetLeft moves with the scroll; its box
      // does not), in the scroller's own coordinates: a column at offsetLeft L is visible from
      // there when scrollLeft = L − pinRight.
      const pinRight = pinned
        ? pinned.getBoundingClientRect().right - el.getBoundingClientRect().left - el.clientLeft
        : 0;
      const want = cell.offsetLeft + cell.offsetWidth - el.clientWidth + 28;
      const lefts = [...(cell.parentElement?.children ?? [])]
        .filter((c) => !(c as HTMLElement).dataset.pin)
        .map((c) => (c as HTMLElement).offsetLeft - pinRight)
        .filter((v) => v >= Math.max(0, want) - 0.5);
      const target = lefts.length ? Math.min(...lefts) : 0;
      el.scrollLeft = Math.max(0, Math.min(target, cell.offsetLeft - pinRight));
    }
    measure();
  };
  const followRef = useRef(follow);
  useEffect(() => {
    followRef.current = follow;
  });

  useEffect(() => {
    followRef.current();
  }, [k, cols]);

  useEffect(() => {
    const el = box.current;
    const table = el?.querySelector('table');
    if (!el || !table || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => followRef.current());
    ro.observe(el);
    ro.observe(table);
    return () => ro.disconnect();
  }, []);

  return (
    <div className={styles.rombergWrap}>
      <div className={styles.rombergCaption}>
        <Formula tex="R_{k,j} = R_{k,j-1} + \frac{R_{k,j-1} - R_{k-1,j-1}}{4^{j} - 1}" />
        <span>
          Digits that agree with the exact integral are in ink. The estimate{' '}
          <Formula tex="R_{k,k}" /> stays pinned; the triangle scrolls.
        </span>
      </div>
      <div
        ref={box}
        className={styles.romberg}
        tabIndex={0}
        role="region"
        aria-label="Romberg table"
        data-more-left={edges.left || undefined}
        data-more-right={edges.right || undefined}
        onScroll={measure}
      >
        <table className={styles.rombergTable}>
          <thead>
            <tr>
              <th scope="col" className={styles.rombergCorner} data-pin="k">
                <Formula tex="k" />
              </th>
              <th scope="col" data-pin="diag" aria-label="estimate R(k, k)">
                <Formula tex="R_{k,k}" />
                <span className={styles.colName}>estimate</span>
              </th>
              {Array.from({ length: cols }, (_, j) => (
                <th key={j} scope="col" aria-label={`column j = ${j}`}>
                  <Formula tex={`j = ${j}`} />
                  <span className={styles.colName}>{COLUMN_NAMES[j] ?? ' '}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trace.map((s, i) => {
              const row = s.info.row as number[];
              const future = i > k;
              return (
                <tr
                  key={i}
                  aria-current={i === k ? 'true' : undefined}
                  data-future={future || undefined}
                  onClick={onSelect ? () => onSelect(i) : undefined}
                >
                  <th scope="row" className={styles.mono} data-pin="k">
                    {i}
                  </th>
                  <td className={styles.mono} data-pin="diag">
                    {future ? null : <Digits v={row[i]} exact={exact} />}
                  </td>
                  {Array.from({ length: cols }, (_, j) => {
                    if (j > i) return <td key={j} />;
                    if (future)
                      return <td key={j} className={styles.mono} aria-label="not yet computed" />;
                    return (
                      <td
                        key={j}
                        className={styles.mono}
                        data-diag={j === i || undefined}
                        data-current={(i === k && j === i) || undefined}
                      >
                        <Digits v={row[j]} exact={exact} />
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
