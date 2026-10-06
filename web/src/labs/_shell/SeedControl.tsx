/**
 * The shared random seed, one control for every seeded lab (global, stochastic, combinatorial):
 * its own "Random seed" rail section, placed directly after Methods (see the rail order in
 * LabShell.tsx). A "Seed" field and a "New seed" button that steps to seed + 1, so every draw is
 * reproducible: the same seed gives the same Mulberry32 stream in TypeScript and in Python.
 *
 *   const [seed, setSeed] = useUrlState('s', 0, codecs.number);
 *   <SeedControl value={seed} onChange={setSeed} />
 */
import type { ReactNode } from 'react';
import { Button } from '../../ui/components/Button';
import { NumberField } from '../../ui/components/NumberField';
import { Tooltip } from '../../ui/components/Tooltip';
import { RailSection } from './LabShell';
import styles from './SeedControl.module.css';

/** The largest seed (Mulberry32 takes a 32-bit unsigned integer). */
export const MAX_SEED = 2 ** 32 - 1;

export function SeedControl({
  value,
  onChange,
  note = 'Every method draws from Mulberry32(seed), bit for bit the stream of the Python reference.',
  section = true,
}: {
  value: number;
  onChange: (seed: number) => void;
  /** One line under the field; `null` hides it. */
  note?: ReactNode;
  /** Render the "Random seed" rail section around the control (default); false: the control only. */
  section?: boolean;
}) {
  const seed = Math.min(MAX_SEED, Math.max(0, Math.round(value) || 0));
  const control = (
    <>
      <div className={styles.row}>
        <NumberField
          label="Seed"
          prefix="Seed"
          value={seed}
          integer
          min={0}
          max={MAX_SEED}
          step={1}
          onChange={(v) => onChange(Math.min(MAX_SEED, Math.max(0, Math.round(v))))}
        />
        <Tooltip content="Use the next seed (seed + 1) and rerun every method" side="top">
          <Button icon="dice" onClick={() => onChange(seed >= MAX_SEED ? 0 : seed + 1)}>
            New seed
          </Button>
        </Tooltip>
      </div>
      {note && <p className={styles.note}>{note}</p>}
    </>
  );
  return section ? <RailSection title="Random seed">{control}</RailSection> : control;
}
