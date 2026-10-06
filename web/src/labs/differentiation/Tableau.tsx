/**
 * The Richardson table D(k, j) (Burden & Faires §4.2): row k starts from the central difference
 * D(k, 0) at h_k; each column j eliminates the h^{2j} term. The diagonal D(k, k) is the estimate.
 * Cells show the error |D(k, j) − f′(x0)| (shaded on a perceptual log scale) or the value; the
 * row under the playhead is highlighted, later rows are faint, and a click seeks to that level.
 */
import { useEffect, useRef } from 'react';
import type { Step } from '../../core/types';
import { sci, sig } from '../../core/format';
import { Formula } from '../../ui/components';
import { SEQUENTIAL, sample } from '../../ui/colors';
import styles from './DifferentiationLab.module.css';

export interface TableauProps {
  trace: readonly Step[];
  k: number;
  exact: number | null;
  show: 'error' | 'value';
  onSelect?: (k: number) => void;
}

export function Tableau({ trace, k, exact, show, onSelect }: TableauProps) {
  const box = useRef<HTMLDivElement>(null);
  const width = Math.max(...trace.map((s) => (s.info.row as unknown[]).length));
  const errs = trace.flatMap((s) =>
    (s.info.row as (number | null)[]).map((v) =>
      exact !== null && v !== null && Number.isFinite(v) ? Math.abs(v - exact) : null,
    ),
  );
  const pos = errs.filter((e): e is number => e !== null && e > 0);
  const lo = pos.length ? Math.log10(Math.min(...pos)) : -16;
  const hi = pos.length ? Math.log10(Math.max(...pos)) : 0;

  useEffect(() => {
    const el = box.current;
    const row = el?.querySelector<HTMLElement>(`[data-row="${k}"]`);
    if (!el || !row) return;
    const head = el.querySelector('thead')?.getBoundingClientRect().height ?? 0;
    if (
      row.offsetTop - head < el.scrollTop ||
      row.offsetTop + row.offsetHeight > el.scrollTop + el.clientHeight
    )
      el.scrollTo({ top: Math.max(0, row.offsetTop - el.clientHeight / 2) });
  }, [k]);

  const shade = (e: number | null) => {
    if (e === null || hi === lo) return undefined;
    // Small errors light, large errors dark: the diagonal brightens as it converges.
    const t = e === 0 ? 1 : 1 - (Math.log10(e) - lo) / (hi - lo);
    const [r, g, b] = sample(SEQUENTIAL, 0.15 + 0.85 * t);
    return `rgba(${r}, ${g}, ${b}, 0.2)`;
  };

  return (
    <div
      ref={box}
      className={styles.tableau}
      tabIndex={0}
      role="region"
      aria-label="Richardson table"
    >
      <table>
        <caption className="visually-hidden">
          Richardson table: rows are levels k, columns are extrapolation steps j; the diagonal is
          the estimate.
        </caption>
        <thead>
          <tr>
            <th scope="col">
              <Formula tex="k" />
            </th>
            <th scope="col">
              <Formula tex="h_k" />
            </th>
            {Array.from({ length: width }, (_, j) => (
              <th key={j} scope="col" title={`Column ${j}: error O(h^${2 * j + 2})`}>
                <Formula tex={`D(k, ${j})`} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {trace.map((s, i) => {
            const row = s.info.row as (number | null)[];
            return (
              <tr
                key={i}
                data-row={i}
                aria-current={i === k ? 'true' : undefined}
                data-future={i > k || undefined}
                onClick={onSelect ? () => onSelect(i) : undefined}
              >
                <td>{i}</td>
                <td>{sci(s.info.h as number, 2)}</td>
                {Array.from({ length: width }, (_, j) => {
                  const v = j < row.length ? row[j] : undefined;
                  if (v === undefined) return <td key={j} />;
                  const e =
                    exact !== null && v !== null && Number.isFinite(v) ? Math.abs(v - exact) : null;
                  return (
                    <td
                      key={j}
                      data-diag={j === row.length - 1 || undefined}
                      style={show === 'error' ? { background: shade(e) } : undefined}
                    >
                      {show === 'error'
                        ? e === null
                          ? '—'
                          : e === 0
                            ? '0'
                            : sci(e, 2)
                        : sig(v, 12)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
