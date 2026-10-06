/**
 * The strip under "This step": one bar per step of the run (a step-size schedule hₖ, FISTA's
 * momentum βₖ, ARC's σₖ, regularized Newton's λₖ), the current step in the method's color, the
 * steps already taken lighter and the steps still to come in neutral gray, with marks under the
 * bars (certified checkpoints, restarts, rejected steps). For AA it shows the step's mixing
 * weights c* instead, as signed bars.
 *
 * The strip has one fixed height for every step and every method, so the panel never moves
 * while the run plays (a window of 64 steps pages forward as the playhead passes it).
 */
import { Formula } from '../../ui/components/Formula';
import { seriesVar } from '../../ui/colors';
import { SciText } from '../../ui/components/Num';
import { short } from './columns';
import type { BarStrip, StripSpec, WeightStrip } from './stepTex';
import { STRIP_WINDOW, barScale, stripWindow } from './strip';
import styles from './UnconstrainedLab.module.css';

function Bars({ spec, k, slot }: { spec: BarStrip; k: number; slot: number }) {
  const n = spec.values.length - 1;
  const [first, last] = stripWindow(n, k);
  const cols = n <= STRIP_WINDOW ? Math.max(1, n) : STRIP_WINDOW;
  const scale = barScale(spec);
  const marks = new Set(spec.marks);
  const color = seriesVar(slot);
  const steps = Array.from({ length: Math.max(0, last - first + 1) }, (_, j) => first + j);
  const refY = spec.ref ? scale(spec.ref.value) : null;
  return (
    <>
      <div className={styles.stripHead}>
        <span className={styles.stripLabel}>
          <Formula tex={spec.label} />
          <span>per step{spec.log ? ', log scale' : ''}</span>
        </span>
        <span className={styles.stripRange}>
          {n > 0 ? `steps ${first}–${last} of ${n}` : 'no steps yet'}
        </span>
      </div>
      <div
        className={styles.stripBars}
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        role="img"
        aria-label={`${spec.label} for steps ${first} to ${last}; step ${k} highlighted.`}
      >
        {steps.map((j) => {
          const v = spec.values[j];
          const h = v === null || !Number.isFinite(v) ? 0 : scale(v);
          const state = j === k ? 'now' : j < k ? 'past' : 'next';
          return (
            <span key={j} className={styles.stripCell}>
              <span
                className={styles.stripBar}
                data-state={state}
                style={{
                  height: `calc((100% - 7px) * ${h.toFixed(4)})`,
                  background: state === 'next' ? undefined : color,
                }}
              />
              {marks.has(j) && <span className={styles.stripMark} data-state={state} />}
            </span>
          );
        })}
        {refY !== null && (
          <span
            className={styles.stripRef}
            style={{ bottom: `calc(7px + (100% - 7px) * ${refY.toFixed(4)})` }}
          />
        )}
      </div>
      <p className={styles.stripLegend}>
        {spec.ref && (
          <span className={styles.stripLegendItem}>
            <span className={styles.stripLegendRef} aria-hidden="true" />
            <Formula tex={spec.ref.tex} />
          </span>
        )}
        {spec.markLabel && spec.marks.length > 0 && (
          <span className={styles.stripLegendItem}>
            <span className={styles.stripLegendDot} aria-hidden="true" />
            {spec.markLabel} ({spec.marks.length})
          </span>
        )}
        <span className={styles.stripLegendItem}>
          <span
            className={styles.stripLegendSwatch}
            style={{ background: color }}
            aria-hidden="true"
          />
          this step
        </span>
      </p>
    </>
  );
}

function Weights({ spec, slot }: { spec: WeightStrip; slot: number }) {
  const w = spec.weights;
  const top = Math.max(1, ...w.map(Math.abs));
  const color = seriesVar(slot);
  const labelled = w.length <= 8;
  // From 4 weights on a ×10ⁿ label is wider than its column: alternate the two label lines.
  const stagger = w.length >= 4;
  return (
    <>
      <div className={styles.stripHead}>
        <span className={styles.stripLabel}>
          <Formula tex="c_i^{\star}" />
          <span>mixing weights (they sum to 1)</span>
        </span>
        <span className={styles.stripRange}>
          {w.length ? `${w.length} iterates` : 'no history yet'}
        </span>
      </div>
      <div
        className={styles.stripWeights}
        style={{ gridTemplateColumns: `repeat(${Math.max(1, w.length)}, minmax(0, 1fr))` }}
        role="img"
        aria-label={
          w.length
            ? `Weights ${w.map((c) => c.toPrecision(3)).join(', ')} on the last ${w.length} iterates.`
            : 'No mixing weights: a plain gradient step.'
        }
      >
        {w.map((c, j) => (
          <span key={j} className={styles.stripWeight}>
            <span
              className={styles.stripWeightBar}
              data-sign={c < 0 ? 'neg' : 'pos'}
              style={{ height: `${((Math.abs(c) / top) * 50).toFixed(2)}%`, background: color }}
            />
          </span>
        ))}
        <span className={styles.stripZero} />
      </div>
      <div
        className={styles.stripWeightLabels}
        style={{ gridTemplateColumns: `repeat(${Math.max(1, w.length)}, minmax(0, 1fr))` }}
        aria-hidden="true"
      >
        {labelled &&
          w.map((c, j) => (
            <span key={j} data-line={stagger && j % 2 === 1 ? '2' : undefined}>
              <SciText text={short(c, 2)} />
            </span>
          ))}
      </div>
      <p className={styles.stripLegend}>
        {w.length ? (
          <span className={styles.stripLegendText}>
            Weights of{' '}
            <Formula tex={`${spec.names[0]}, \\dots, ${spec.names[spec.names.length - 1]}`} />{' '}
            (oldest first); a negative weight extrapolates beyond the history.
          </span>
        ) : (
          <span className={styles.stripLegendText}>
            A plain gradient step: there is no history to mix yet.
          </span>
        )}
      </p>
    </>
  );
}

export function StepStrip({ spec, k, slot }: { spec: StripSpec; k: number; slot: number }) {
  return (
    <div className={styles.strip}>
      {spec.kind === 'bars' ? (
        <Bars spec={spec} k={k} slot={slot} />
      ) : (
        <Weights spec={spec} slot={slot} />
      )}
    </div>
  );
}
