import styles from './Controls.module.css';

export interface ToggleProps {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  /** Render the label visibly next to the switch. */
  showLabel?: boolean;
  disabled?: boolean;
}

export function Toggle({ checked, onChange, label, showLabel = false, disabled }: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={showLabel ? undefined : label}
      disabled={disabled}
      className={styles.toggle}
      onClick={() => onChange(!checked)}
    >
      <span className={styles.switch} aria-hidden="true" />
      {showLabel && <span>{label}</span>}
    </button>
  );
}
