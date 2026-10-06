/**
 * "This step" in the MethodCard: the focused method's update for the step under the playhead
 * with its numbers filled in, one sentence on what the method decided, and the step's
 * quantities.
 *
 * The block keeps one layout for the whole run. Each part (formula, note, quantities) is a grid
 * cell that also holds, hidden, the same part of the largest steps of the run (`layoutSamples`),
 * so every part has the height of its tallest version from the first frame and nothing in the
 * block or below it moves while the trace plays. Every formula of the run shares one type size
 * (FitGroup). At k = 0 the start note stands in the formula's cell, top-aligned.
 */
import { memo, useMemo, type ReactNode } from 'react';
import { Formula } from '../../ui/components/Formula';
import { StepBlock } from '../_shell';
import { FitBox, FitGroup } from './FitMath';
import { MathText } from './MathText';
import { layoutSamples, stepView, type Quantity, type StepInput, type StepView } from './stepTex';
import { StepStrip } from './StepStrip';
import styles from './UnconstrainedLab.module.css';

const SCROLL_LABEL = 'This step’s update, scrolls sideways';

/** One id per run (trace): resets the block's kept height when a new run starts. */
const RUN_IDS = new WeakMap<object, string>();
let nextRun = 0;
function runId(trace: object): string {
  let id = RUN_IDS.get(trace);
  if (!id) RUN_IDS.set(trace, (id = `unc-step-${nextRun++}`));
  return id;
}

/** A part of the block: the current version and, hidden in the same cell, the samples'. */
function Stack({ ghosts, children }: { ghosts: ReactNode[]; children: ReactNode }) {
  return (
    <div className={styles.stepStack}>
      {ghosts.map((g, i) => (
        <div key={i} className={styles.stepPart} data-ghost="" aria-hidden="true">
          {g}
        </div>
      ))}
      <div className={styles.stepPart}>{children}</div>
    </div>
  );
}

/** The formula of a step, or the start note at k = 0 (top-aligned in the formula's cell). */
const Head = memo(function Head({ v }: { v: StepView }) {
  return v.tex ? (
    <FitBox tex={v.tex} className={styles.stepEqTex} label={SCROLL_LABEL} />
  ) : (
    <p className={styles.stepEqNote}>
      <MathText text={v.note} />
    </p>
  );
});

/** The note of a step (none at k = 0: the start note is in the formula's cell). */
const Note = memo(function Note({ v }: { v: StepView }) {
  return v.tex && v.note ? (
    <p className={styles.stepEqNote}>
      <MathText text={v.note} />
    </p>
  ) : null;
});

const Grid = memo(function Grid({ q }: { q: readonly Quantity[] }) {
  return q.length > 0 ? (
    <dl className={styles.live}>
      {q.map((c) => (
        <div key={c.tex} className={styles.liveCell} data-wide={c.wide || undefined}>
          <dt className={styles.liveLabel}>
            <Formula tex={c.tex} />
          </dt>
          <dd className={styles.liveValue}>{c.value}</dd>
        </div>
      ))}
    </dl>
  ) : null;
});

export function StepEquation(props: StepInput & { slot: number }) {
  const { kind, method, trace, g, params, problem, slot } = props;
  // Once per run (method, trace, parameters), not per step.
  const samples = useMemo(
    () =>
      layoutSamples({ kind, method, trace, params, problem }).map((s) =>
        stepView({ kind, method, trace, g: s, params, problem }),
      ),
    [kind, method, trace, params, problem],
  );
  const v = stepView({ kind, method, trace, g, params, problem });
  return (
    // Step g − 1 → g: the update that produced the iterate under the playhead.
    <StepBlock k={g - 1} label={g === 0 ? 'Start point' : undefined} reset={runId(trace)}>
      <FitGroup run={trace} className={styles.stepParts}>
        {v.strip && <StepStrip spec={v.strip} k={g} slot={slot} />}
        <Stack
          ghosts={samples.map((s, i) => (
            <Head key={i} v={s} />
          ))}
        >
          <Head v={v} />
        </Stack>
        <Stack
          ghosts={samples.map((s, i) => (
            <Note key={i} v={s} />
          ))}
        >
          <Note v={v} />
        </Stack>
        <Stack
          ghosts={samples.map((s, i) => (
            <Grid key={i} q={s.quantities} />
          ))}
        >
          <Grid q={v.quantities} />
        </Stack>
      </FitGroup>
    </StepBlock>
  );
}
