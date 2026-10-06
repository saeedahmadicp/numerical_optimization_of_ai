import { useCallback, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import styles from './Controls.module.css';

export interface SliderProps {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  /** Logarithmic mapping (requires min > 0). */
  log?: boolean;
  /** Linear step (ignored for log sliders, which move by `logSteps` per decade). */
  step?: number;
  integer?: boolean;
  /** Keyboard steps per decade on a log slider. */
  logSteps?: number;
  disabled?: boolean;
  label: string;
  /** Human text for screen readers, e.g. `1×10⁻⁸`. */
  valueText?: string;
  onCommit?: (v: number) => void;
  className?: string;
}

const LOG_POSITIONS = 1000;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Accessible slider (role="slider") with log-scale support and full keyboard control. */
export function Slider({
  value,
  min,
  max,
  onChange,
  log = false,
  step,
  integer = false,
  logSteps = 10,
  disabled,
  label,
  valueText,
  onCommit,
  className,
}: SliderProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const isLog = log && min > 0 && max > min;

  const toPos = useCallback(
    (v: number) => {
      const c = clamp(v, min, max);
      if (isLog) return Math.log(c / min) / Math.log(max / min);
      return max === min ? 0 : (c - min) / (max - min);
    },
    [min, max, isLog],
  );
  const fromPos = useCallback(
    (u: number) => {
      const p = clamp(u, 0, 1);
      let v = isLog ? min * (max / min) ** p : min + p * (max - min);
      if (integer) v = Math.round(v);
      else if (!isLog && step) v = Math.round((v - min) / step) * step + min;
      else if (isLog) {
        // Snap to 3 significant figures so values read cleanly (1.23e-4, not 1.2297e-4).
        v = Number(v.toPrecision(3));
      }
      return clamp(v, min, max);
    },
    [min, max, isLog, integer, step],
  );

  const setFromClient = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r || r.width === 0) return;
    onChange(fromPos((clientX - r.left) / r.width));
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (disabled || e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
    setFromClient(e.clientX);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragging) setFromClient(e.clientX);
  };
  const onPointerUp = () => {
    if (!dragging) return;
    setDragging(false);
    onCommit?.(value);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const big = e.key === 'PageUp' || e.key === 'PageDown' || e.shiftKey;
    let next: number | null = null;
    const dir =
      e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'PageUp'
        ? 1
        : e.key === 'ArrowLeft' || e.key === 'ArrowDown' || e.key === 'PageDown'
          ? -1
          : 0;
    if (dir !== 0) {
      if (isLog) {
        const k = big ? 1 : 1 / logSteps; // decades
        next = clamp(Number((value * 10 ** (dir * k)).toPrecision(3)), min, max);
      } else {
        const s = step ?? (integer ? 1 : (max - min) / 100);
        next = clamp(value + dir * s * (big ? 10 : 1), min, max);
        if (integer) next = Math.round(next);
      }
    } else if (e.key === 'Home') next = min;
    else if (e.key === 'End') next = max;
    if (next !== null) {
      e.preventDefault();
      e.stopPropagation();
      onChange(next);
      onCommit?.(next);
    }
  };

  const pct = toPos(value) * 100;
  // ARIA numbers must be plain decimals: a log slider exposes its position (0–1000) and puts the
  // real value in aria-valuetext ("1e-12" is not a valid aria-valuemin).
  const plain = (n: number) => !/e/i.test(String(n));
  const aria =
    isLog || !plain(min) || !plain(max) || !plain(value)
      ? {
          min: 0,
          max: LOG_POSITIONS,
          now: Math.round(toPos(value) * LOG_POSITIONS),
          text: valueText ?? String(value),
        }
      : { min, max, now: value, text: valueText };
  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={label}
      aria-valuemin={aria.min}
      aria-valuemax={aria.max}
      aria-valuenow={aria.now}
      aria-valuetext={aria.text}
      aria-disabled={disabled || undefined}
      aria-orientation="horizontal"
      data-dragging={dragging}
      className={`${styles.slider} ${className ?? ''}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
    >
      <div className={styles.track}>
        <div className={styles.fill} style={{ width: `${pct}%` }} />
      </div>
      <div className={styles.thumb} style={{ left: `${pct}%` }} />
    </div>
  );
}
