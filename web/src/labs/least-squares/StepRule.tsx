/**
 * "This step" in the MethodCard (shared StepBlock): the update rule of the focused method with
 * the numbers of the step under the playhead substituted — the step taken from 𝐱ₖ (trace[k]) to
 * 𝐱ₖ₊₁ (trace[k + 1]); at the end of the run, the last iterate.
 */
import { useEffect, useState } from 'react';
import type { Step } from '../../core/types';
import { int, sig, vec } from '../../core/format';
import { Formula } from '../../ui/components';
import { StepBlock } from '../_shell';
import { roundingNoise, stepRuleTex } from './text';
import type { MethodKind } from './models';
import styles from './LeastSquaresLab.module.css';

export interface StepRuleProps {
  kind: MethodKind;
  trace: readonly Step[];
  k: number;
  /** The two parameter names (a, b; V, K; x, y). */
  names: readonly [string, string];
}

/**
 * A counter that ticks when web fonts finish loading. `<Formula fit>` measures the formula when
 * its HTML or its box changes; KaTeX's fonts arriving later widen the glyphs without either, so
 * the formula is re-mounted (and measured again) on every tick.
 */
function useFontEpoch(): number {
  const [epoch, setEpoch] = useState(0);
  useEffect(() => {
    const fonts = typeof document !== 'undefined' ? document.fonts : undefined;
    if (!fonts) return;
    let alive = true;
    const bump = () => {
      if (alive) setEpoch((e) => e + 1);
    };
    void fonts.ready.then(bump);
    fonts.addEventListener('loadingdone', bump);
    return () => {
      alive = false;
      fonts.removeEventListener('loadingdone', bump);
    };
  }, []);
  return epoch;
}

export function StepRule({ kind, trace, k, names }: StepRuleProps) {
  const epoch = useFontEpoch();
  const tex = stepRuleTex(kind, trace, k, names);
  if (!tex) return null;
  const run = runId(trace);
  const s = trace[k];
  // A component that cancelled to rounding level makes f and ∇f noise (the rule says so).
  const noise = roundingNoise(s.x as number[], trace[k - 1]?.x as number[] | undefined).length > 0;
  const cells: { tex: string; value: string; wide?: boolean }[] = [
    { tex: 'k', value: int(k) },
    { tex: 'f(\\mathbf{x}_k)', value: noise ? 'noise' : sig(s.fun, 5) },
    { tex: '\\|\\nabla f(\\mathbf{x}_k)\\|_2', value: noise ? 'noise' : sig(s.gradNorm, 3) },
    { tex: '\\mathbf{x}_k', value: vec(s.x as number[], 5), wide: true },
  ];
  return (
    <StepBlock k={k} last={k >= trace.length - 1} reset={run}>
      <div className={styles.stepRuleBody}>
        {/* One type size for every step of the run (the rule changes shape between steps). */}
        <Formula key={epoch} tex={tex} display fit fitGroup={run} />
      </div>
      <dl className={styles.live}>
        {cells.map((c) => (
          <div key={c.tex} className={styles.liveCell} data-wide={c.wide || undefined}>
            <dt className={styles.liveLabel}>
              <Formula tex={c.tex} />
            </dt>
            <dd className={styles.liveValue}>{c.value}</dd>
          </div>
        ))}
      </dl>
    </StepBlock>
  );
}

/** One id per run (trace): the fit group of its formulas and the reset of the block height. */
const RUN_IDS = new WeakMap<object, string>();
let nextRun = 0;
function runId(trace: object): string {
  let id = RUN_IDS.get(trace);
  if (!id) RUN_IDS.set(trace, (id = `ls-step-${nextRun++}`));
  return id;
}
