/**
 * "This step" in the MethodCard: the focused method's rule with this step's numbers, one
 * sentence on what happened, and the step's quantities with exact labels.
 *
 * The formula and its note change shape from step to step (an accepted proposal needs fewer rows
 * than a rejected one). Each part also holds, hidden in its grid cell, the same part of the
 * tallest steps of the run (`layoutSamples`), so every part keeps one height for the whole run
 * and nothing in the block or below it moves. The formulas of a run share one type size.
 */
import { Fragment, useMemo, type ReactNode } from 'react';
import { Formula } from '../../ui/components/Formula';
import { StepBlock } from '../_shell';
import { cellSpans, layoutSamples, stepQuantities, stepTex, type StepTexInput } from './stepTex';
import { useFontsEpoch } from './useFontsEpoch';
import styles from './GlobalLab.module.css';

type StepTexOut = ReturnType<typeof stepTex>;

/** A note in prose with inline TeX between $…$. */
export function MathNote({ text }: { text: string }) {
  const parts = text.split('$');
  return (
    <>
      {parts.map((p, i) =>
        i % 2 === 1 ? <Formula key={i} tex={p} /> : <Fragment key={i}>{p}</Fragment>,
      )}
    </>
  );
}

const START_NOTE =
  'Step 0 is the start: the first point or population and its values. Step forward to see the first update.';

/** One id per run (trace): the fit group of its formulas and the reset of the block height. */
const RUN_IDS = new WeakMap<object, string>();
let nextRun = 0;
function runId(trace: object): string {
  let id = RUN_IDS.get(trace);
  if (!id) RUN_IDS.set(trace, (id = `global-step-${nextRun++}`));
  return id;
}

/** The formula of a step, or the start note at k = 0 (top-aligned in the formula's cell). */
function Head({
  r,
  fonts,
  group,
}: {
  r: StepTexOut | null;
  fonts: number;
  /** One type size for every formula of the run (the samples included). */
  group: string;
}) {
  return r ? (
    <div className={styles.stepEqTex}>
      {/* KaTeX fonts change the width of a formula: re-fit once they are in. */}
      <Formula key={fonts} tex={r.tex} display fit fitGroup={group} />
    </div>
  ) : (
    <p className={styles.stepEqNote}>{START_NOTE}</p>
  );
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

const noteOf = (r: StepTexOut | null) =>
  r ? (
    <p className={styles.stepEqNote}>
      <MathNote text={r.note} />
    </p>
  ) : null;

export function StepEquation(props: StepTexInput & { popSize?: number }) {
  const { method, trace, params, n, k, popSize } = props;
  const q = stepQuantities(method, trace[k], popSize);
  const spans = cellSpans(q.map((c) => !!c.wide));
  const fonts = useFontsEpoch();
  // Once per run (method, trace, parameters), not per step.
  const samples = useMemo(
    () =>
      layoutSamples({ method, trace, params, n })
        .filter((s) => s < trace.length)
        .map((s) => stepTex({ method, trace, params, n, k: s })),
    [method, trace, params, n],
  );
  const run = `${runId(trace)}-${method}`;
  const shown = stepTex({ method, trace, params, n, k });
  return (
    // Step k − 1 → k: the update that produced the step under the playhead.
    <StepBlock k={k - 1} label={k === 0 ? 'Start point' : undefined} reset={run}>
      <Stack
        ghosts={samples.map((r, i) => (
          <Head key={i} r={r} fonts={fonts} group={run} />
        ))}
      >
        <Head r={shown} fonts={fonts} group={run} />
      </Stack>
      <Stack ghosts={samples.map(noteOf)}>{noteOf(shown)}</Stack>
      {q.length > 0 && (
        <dl className={styles.live}>
          {q.map((c, i) => (
            <div
              key={c.tex}
              className={styles.liveCell}
              style={spans[i] > 1 ? { gridColumn: `span ${spans[i]}` } : undefined}
            >
              <dt className={styles.liveLabel}>
                <Formula tex={c.tex} />
              </dt>
              <dd className={styles.liveValue}>{c.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </StepBlock>
  );
}
