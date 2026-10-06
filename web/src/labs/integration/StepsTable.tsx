/** The iteration table of the focused run, synced to the playhead (columns: stepColumns.ts). */
import type { Step } from '../../core/types';
import { TableView } from '../../viz';
import { columnsFor } from './stepColumns';
import styles from './IntegrationLab.module.css';

export function StepsTable({
  method,
  name,
  trace,
  k,
  onSelect,
}: {
  method: string;
  name: string;
  trace: readonly Step[];
  k: number;
  onSelect: (k: number) => void;
}) {
  // The wrapper (display: contents) only narrows the cell padding so all columns fit the
  // details column.
  return (
    <div className={styles.steps}>
      <TableView
        columns={columnsFor(method)}
        rows={trace}
        highlight={k}
        futureAfter={k}
        onSelect={onSelect}
        ariaLabel={`${name}: steps`}
        maxHeight="100%"
        rowKey={(s) => s.k}
      />
    </div>
  );
}
