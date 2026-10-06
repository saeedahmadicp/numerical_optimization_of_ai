import { useEffect, useState, type KeyboardEvent } from 'react';
import { MINUS } from '../../core/format';
import styles from './Controls.module.css';

export interface NumberFieldProps {
  value: number;
  onChange: (v: number) => void;
  label: string;
  /** Visible short prefix inside the field, e.g. `x₀`. */
  prefix?: string;
  min?: number | null;
  max?: number | null;
  integer?: boolean;
  /** Arrow-key increment (multiplied by 10 with Shift). For log fields, steps are ×/÷ 10^(1/10). */
  step?: number;
  log?: boolean;
  /** Editable text (while the field has focus): what a person types, e.g. `1e-8`. */
  format?: (v: number) => string;
  /** Text while the field is not focused, e.g. typeset `1×10⁻⁸` (default: `format`). */
  display?: (v: number) => string;
  className?: string;
  width?: number | string;
}

const FROM_SUP: Record<string, string> = {
  '⁰': '0',
  '¹': '1',
  '²': '2',
  '³': '3',
  '⁴': '4',
  '⁵': '5',
  '⁶': '6',
  '⁷': '7',
  '⁸': '8',
  '⁹': '9',
  '⁻': '-',
};

/** Accepts `1e-8`, `1×10⁻⁸`, `1x10^-8`, `−3` and `100,000`. */
const parse = (s: string) =>
  Number(
    s
      .trim()
      .replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]/g, (c) => FROM_SUP[c])
      .replaceAll(MINUS, '-')
      .replace(/,/g, '')
      .replace(/[×x]10\^?/i, 'e'),
  );

/** A compact numeric input: accepts `1e-8`, commits on Enter/blur, ↑/↓ to nudge, Esc to revert. */
export function NumberField({
  value,
  onChange,
  label,
  prefix,
  min = null,
  max = null,
  integer,
  step,
  log,
  format,
  display,
  className,
  width,
}: NumberFieldProps) {
  const show = (v: number) =>
    format ? format(v) : Number.isFinite(v) ? String(Number(v.toPrecision(6))) : '';
  const shown = (v: number) => (display ? display(v) : show(v));
  const [text, setText] = useState(() => shown(value));
  const [focused, setFocused] = useState(false);
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    // Sync external value changes (e.g. clicking the plot sets x₀) while not editing.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (!focused) setText(shown(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, focused]);

  const valid = (v: number) =>
    Number.isFinite(v) &&
    (min === null || v >= min) &&
    (max === null || v <= max) &&
    (!integer || Number.isInteger(v));

  const commit = (editing = false) => {
    const v = parse(text);
    if (valid(v)) {
      setInvalid(false);
      if (v !== value) onChange(v);
      setText(editing ? show(v) : shown(v));
    } else {
      setInvalid(true);
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      commit(true);
      (e.target as HTMLInputElement).select();
    } else if (e.key === 'Escape') {
      setText(show(value));
      setInvalid(false);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const dir = e.key === 'ArrowUp' ? 1 : -1;
      const base = valid(parse(text)) ? parse(text) : value;
      let next = log
        ? base * 10 ** (dir * (e.shiftKey ? 1 : 0.1))
        : base + dir * (step ?? (integer ? 1 : 0.1)) * (e.shiftKey ? 10 : 1);
      next = Number(next.toPrecision(log ? 3 : 10));
      if (min !== null) next = Math.max(min, next);
      if (max !== null) next = Math.min(max, next);
      if (integer) next = Math.round(next);
      setText(show(next));
      setInvalid(false);
      onChange(next);
    }
  };

  return (
    <label
      className={`${styles.field} ${className ?? ''}`}
      data-invalid={invalid}
      style={{ width }}
    >
      {prefix && (
        <span className={styles.fieldPrefix} aria-hidden="true">
          {prefix}
        </span>
      )}
      <input
        className={styles.input}
        aria-label={label}
        aria-invalid={invalid}
        inputMode="decimal"
        spellCheck={false}
        autoComplete="off"
        value={text}
        onFocus={(e) => {
          setFocused(true);
          // Editing uses plain notation (`1e-8`); the typeset text returns on blur.
          if (!invalid) setText(show(value));
          // Select on the next frame (after the browser places the caret), but only if the field
          // still has focus: `select()` focuses its input, so a Tab (or Escape) pressed before the
          // frame would otherwise pull focus back to a field the viewer has already left.
          const input = e.target;
          requestAnimationFrame(() => {
            if (document.activeElement === input) input.select();
          });
        }}
        onBlur={() => {
          setFocused(false);
          commit();
        }}
        onChange={(e) => {
          setText(e.target.value);
          setInvalid(false);
        }}
        onKeyDown={onKeyDown}
      />
    </label>
  );
}
