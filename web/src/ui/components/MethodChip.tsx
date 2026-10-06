import { seriesVar } from '../colors';
import { Icon } from './Icon';
import styles from './MethodChip.module.css';

export interface MethodChipProps {
  name: string;
  /** Series slot (0-based) → `--series-{slot+1}`. */
  slot: number;
  onRemove?: () => void;
  /**
   * Let the name wrap to this many lines (2 in the rail's method slots, so a long name is never
   * cut at desktop width); the chip then grows to fit and keeps a rounded-rectangle shape.
   */
  lines?: 2;
}

/** Color swatch + method name (+ remove). The name always travels with the color. */
export function MethodChip({ name, slot, onRemove, lines }: MethodChipProps) {
  return (
    <span
      className={`${styles.chip} ${onRemove ? '' : styles.static}`}
      data-lines={lines}
      style={{ ['--_c' as string]: seriesVar(slot) }}
      title={lines ? name : undefined}
    >
      <span className={styles.swatch} aria-hidden="true" />
      <span className={styles.name}>{name}</span>
      {onRemove && (
        <button
          type="button"
          className={styles.remove}
          aria-label={`Remove ${name}`}
          onClick={onRemove}
        >
          <Icon name="x" size={12} />
        </button>
      )}
    </span>
  );
}

/** Just the swatch (for legends and option rows). */
export function Swatch({ slot, size = 10 }: { slot: number; size?: number }) {
  return (
    <span
      aria-hidden="true"
      className={styles.swatch}
      style={{ ['--_c' as string]: seriesVar(slot), width: size, height: size }}
    />
  );
}
