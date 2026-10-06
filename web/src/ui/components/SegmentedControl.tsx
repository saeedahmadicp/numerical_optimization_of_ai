import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react';
import { motion } from 'motion/react';
import { Tooltip } from './Tooltip';
import { useMotionTransitions } from '../motion';
import styles from './Controls.module.css';

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  /** Accessible name when `label` is an icon. */
  ariaLabel?: string;
  /** Hover/focus tip (e.g. the method name next to a color swatch). */
  tooltip?: string;
}

export interface SegmentedControlProps<T extends string> {
  value: T;
  options: SegmentOption<T>[];
  onChange: (v: T) => void;
  label: string;
  className?: string;
  fullWidth?: boolean;
}

/** Radio group styled as a segmented control (arrow keys move the selection). */
export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  label,
  className,
  fullWidth,
}: SegmentedControlProps<T>) {
  const group = useId();
  const motionT = useMotionTransitions();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const idx = options.findIndex((o) => o.value === value);

  const onKeyDown = (e: KeyboardEvent) => {
    const d =
      e.key === 'ArrowRight' || e.key === 'ArrowDown'
        ? 1
        : e.key === 'ArrowLeft' || e.key === 'ArrowUp'
          ? -1
          : 0;
    if (!d) return;
    e.preventDefault();
    e.stopPropagation();
    const next = (idx + d + options.length) % options.length;
    onChange(options[next].value);
    refs.current[next]?.focus();
  };

  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={`${styles.segmented} ${className ?? ''}`}
      style={fullWidth ? { display: 'flex', width: '100%' } : undefined}
      onKeyDown={onKeyDown}
    >
      {options.map((o, i) => {
        const selected = o.value === value;
        const button = (
          <button
            key={o.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={o.ariaLabel}
            tabIndex={selected || (idx < 0 && i === 0) ? 0 : -1}
            className={styles.segment}
            onClick={() => onChange(o.value)}
          >
            {selected && (
              <motion.span
                layoutId={`seg-${group}`}
                className={styles.segmentPill}
                transition={motionT.base}
              />
            )}
            {o.label}
          </button>
        );
        return o.tooltip ? (
          <Tooltip key={o.value} content={o.tooltip} describe={o.tooltip !== o.ariaLabel}>
            {button}
          </Tooltip>
        ) : (
          button
        );
      })}
    </div>
  );
}
