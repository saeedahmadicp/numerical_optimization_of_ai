/**
 * A readout that keeps one width for the whole run: the widest value of the run is laid out
 * invisibly in the same grid cell as the current one, so a value that grows during playback
 * ("N = 2 · h = 2" → "N = 512 · h = 0.007813") never moves its neighbors.
 */
import type { ReactNode } from 'react';

export function Reserve({
  widest: ghosts,
  children,
  align = 'start',
  className,
}: {
  /** The widest values of the run (all laid out invisibly; the cell takes the largest). */
  widest: ReactNode | readonly ReactNode[];
  children: ReactNode;
  align?: 'start' | 'end';
  className?: string;
}) {
  const list: readonly ReactNode[] = Array.isArray(ghosts) ? ghosts : [ghosts];
  // The class goes on an outer span, so a lab's CSS can still hide the readout (display: none).
  return (
    <span className={className}>
      <span style={{ display: 'inline-grid', justifyItems: align }}>
        {list.map((g, i) => (
          <span key={i} aria-hidden="true" style={{ gridArea: '1 / 1', visibility: 'hidden' }}>
            {g}
          </span>
        ))}
        <span style={{ gridArea: '1 / 1' }}>{children}</span>
      </span>
    </span>
  );
}
