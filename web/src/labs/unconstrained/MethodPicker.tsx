/**
 * The lab's method selection: the shared MethodSlots (chips, colors, parameter controls) for the
 * selected methods, and an "Add a method" picker grouped by family — gradient, momentum,
 * adaptive, Newton, quasi-Newton, conjugate gradient, trust region, derivative-free — because a
 * flat list of 34 names hides the structure a reader navigates by.
 */
import type { RegisteredMethod } from '../../core/registry';
import { Select, Swatch, type SelectOption } from '../../ui/components';
import { MethodSlots, firstFreeSlot, type MethodSelection } from '../_shell';
import { GROUPS } from './catalog';
import styles from './UnconstrainedLab.module.css';

export function MethodPicker({
  available,
  value,
  onChange,
}: {
  available: readonly RegisteredMethod[];
  value: readonly MethodSelection[];
  onChange: (v: MethodSelection[]) => void;
}) {
  const byId = new Map(available.map((m) => [m.spec.id, m]));
  const chosen = new Set(value.map((s) => s.id));
  // MethodSlots shows no add control when every method it is given is selected.
  const selected = available.filter((m) => chosen.has(m.spec.id));
  const free = firstFreeSlot(value);
  const grouped = GROUPS.flatMap((g) => g.methods.map((id) => ({ id, group: g.label })));
  const known = new Set(grouped.map((x) => x.id));
  const rest = available
    .filter((m) => !known.has(m.spec.id))
    .map((m) => ({ id: m.spec.id, group: 'Other' }));
  const options: SelectOption[] = [...grouped, ...rest]
    .filter(({ id }) => byId.has(id) && !chosen.has(id))
    .map(({ id, group }) => {
      const m = byId.get(id)!;
      return {
        value: id,
        label: m.spec.name,
        description: m.spec.summary,
        group,
        keywords: `${group} ${m.spec.tags?.join(' ') ?? ''}`,
      };
    });
  return (
    <div className={styles.picker}>
      <MethodSlots available={selected} value={value} onChange={onChange} />
      {free >= 0 && options.length > 0 && (
        <Select
          label="Add a method"
          value={null}
          placeholder="Add a method to compare…"
          options={options}
          searchable
          renderValue={() => (
            <span className={styles.addValue}>
              <Swatch slot={free} />
              Add a method to compare…
            </span>
          )}
          onChange={(id) => onChange([...value, { id, slot: free, params: {} }])}
        />
      )}
    </div>
  );
}
