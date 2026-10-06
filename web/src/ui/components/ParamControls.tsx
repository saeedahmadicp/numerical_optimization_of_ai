import type { ParamSpec, ParamValue, Params } from '../../core/types';
import { int, sci, sig } from '../../core/format';
import { choiceLabel } from '../../core/registry';
import { Slider } from './Slider';
import { NumberField } from './NumberField';
import { Toggle } from './Toggle';
import { SegmentedControl } from './SegmentedControl';
import { Select } from './Select';
import { Tooltip } from './Tooltip';
import { Icon } from './Icon';
import { Formula } from './Formula';
import styles from './ParamControls.module.css';

/**
 * The most label characters (all options together) a segmented choice shows in the 272 px rail:
 * "QR · Normal equations · SVD" (21) fits with room to spare; longer sets become a menu.
 */
const SEGMENT_CHARS = 30;

export interface ParamControlsProps {
  specs: readonly ParamSpec[];
  values: Params;
  onChange: (name: string, value: ParamValue) => void;
  /** Accessible prefix, e.g. the method name ("Gradient descent: alpha"). */
  labelPrefix?: string;
  /** Hide these params (e.g. `max_iter` when the lab manages it). */
  hidden?: readonly string[];
}

const fmt = (spec: ParamSpec, v: number) =>
  spec.log ? sci(v, 3) : spec.kind === 'int' ? int(v) : sig(v, 4);

/** Editable text for a number field: e-notation outside [1e-3, 1e5) (`1e-6`, not `0.000001`). */
const fieldText = (v: number) => {
  if (!Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-3 || a >= 1e5))
    return v
      .toExponential(3)
      .replace(/\.?0+e/, 'e')
      .replace('e+', 'e');
  return String(Number(v.toPrecision(6)));
};

/**
 * Text shown while the field is not being edited: typeset notation (`1×10⁻¹⁰`, `100,000`).
 * While it has focus the field shows `fieldText` (`1e-10`), which is what a person types.
 */
const displayText = (spec: ParamSpec) => (v: number) => {
  if (!Number.isFinite(v)) return '';
  const a = Math.abs(v);
  if (v !== 0 && (a < 1e-3 || a >= 1e5)) return spec.kind === 'int' ? int(v) : sci(v, 4);
  return spec.kind === 'int' ? int(v) : String(Number(v.toPrecision(6))).replace(/^-/, '−');
};

/** `max_iter` → `Max iter` (used when a ParamSpec has no `label`). */
const humanize = (name: string) => {
  const s = name.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};

/** Default slider range when a ParamSpec leaves min/max open. */
function range(spec: ParamSpec): [number, number] {
  const d = Number(spec.default);
  let lo = spec.min ?? (spec.log ? d / 1000 : d >= 0 ? 0 : d * 10);
  let hi = spec.max ?? (spec.log ? d * 1000 : d === 0 ? 1 : Math.abs(d) * 10);
  if (spec.log && lo <= 0) lo = Math.min(hi, d) / 1e6;
  if (hi <= lo) hi = lo + 1;
  return [lo, hi];
}

/** Builds one control per ParamSpec: slider+field (float/int), switch (bool), segments/select (choice). */
export function ParamControls({
  specs,
  values,
  onChange,
  labelPrefix,
  hidden = [],
}: ParamControlsProps) {
  return (
    <div className={styles.list}>
      {specs
        .filter((s) => !hidden.includes(s.name))
        .map((spec) => {
          const value = values[spec.name] ?? spec.default;
          const display = spec.label ?? humanize(spec.name);
          const label = labelPrefix ? `${labelPrefix}: ${display}` : display;
          const isDefault = JSON.stringify(value) === JSON.stringify(spec.default);
          const name = (
            <span className={styles.name}>
              {spec.tex && (
                <span className={styles.tex} aria-hidden="true">
                  <Formula tex={spec.tex} />
                </span>
              )}
              <span className={styles.label}>{display}</span>
              {spec.help && (
                <Tooltip content={spec.help} toggle describe={false}>
                  <button
                    type="button"
                    className={styles.help}
                    aria-label={`About ${display}: ${spec.help}`}
                  >
                    <Icon name="info" size={13} />
                  </button>
                </Tooltip>
              )}
            </span>
          );
          const resetBtn = !isDefault && (
            <button
              type="button"
              className={styles.reset}
              onClick={() => onChange(spec.name, spec.default)}
              aria-label={`Reset ${label}`}
            >
              Reset
            </button>
          );

          if (spec.kind === 'bool') {
            return (
              <div key={spec.name} className={`${styles.row} ${styles.inline}`}>
                {name}
                <Toggle
                  checked={Boolean(value)}
                  onChange={(v) => onChange(spec.name, v)}
                  label={label}
                />
              </div>
            );
          }
          if (spec.kind === 'choice') {
            const options = spec.choices.map((c) => ({ value: c, label: choiceLabel(spec, c) }));
            // Segments only when every label fits the rail at its full width; otherwise a menu
            // (a label is never cut, and the row never wraps).
            const segmented =
              spec.choices.length <= 3 &&
              options.reduce((n, o) => n + o.label.length, 0) <= SEGMENT_CHARS;
            return (
              <div key={spec.name} className={styles.row}>
                <div className={styles.head}>
                  {name}
                  {resetBtn}
                </div>
                {segmented ? (
                  <SegmentedControl
                    fullWidth
                    label={label}
                    value={String(value)}
                    options={options}
                    onChange={(v) => onChange(spec.name, v)}
                  />
                ) : (
                  <Select
                    label={label}
                    value={String(value)}
                    options={options}
                    onChange={(v) => onChange(spec.name, v)}
                  />
                )}
              </div>
            );
          }
          if (spec.kind === 'vector') {
            const vec = (Array.isArray(value) ? value : [value]) as number[];
            return (
              <div key={spec.name} className={styles.row}>
                <div className={styles.head}>
                  {name}
                  {resetBtn}
                </div>
                <div className={styles.vector}>
                  {vec.map((v, i) => (
                    <NumberField
                      key={i}
                      label={`${label}[${i}]`}
                      value={v}
                      onChange={(nv) =>
                        onChange(
                          spec.name,
                          vec.map((x, j) => (j === i ? nv : x)),
                        )
                      }
                    />
                  ))}
                </div>
              </div>
            );
          }
          const [lo, hi] = range(spec);
          const num = Number(value);
          const integer = spec.kind === 'int';
          return (
            <div key={spec.name} className={styles.row}>
              <div className={styles.head}>
                {name}
                {resetBtn}
              </div>
              <div className={styles.control}>
                <Slider
                  label={label}
                  value={num}
                  min={lo}
                  max={hi}
                  log={spec.log}
                  integer={integer}
                  valueText={fmt(spec, num)}
                  onChange={(v) => onChange(spec.name, v)}
                />
                <NumberField
                  className={styles.num}
                  label={label}
                  value={num}
                  min={spec.min}
                  max={spec.max}
                  integer={integer}
                  log={spec.log}
                  format={fieldText}
                  display={displayText(spec)}
                  onChange={(v) => onChange(spec.name, v)}
                />
              </div>
            </div>
          );
        })}
    </div>
  );
}
