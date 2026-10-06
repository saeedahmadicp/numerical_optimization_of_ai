/**
 * "This step": the update rule with the numbers of the current step filled in (the linear system
 * Newton solved, or Broyden's secant pair), and the model error of a Broyden matrix.
 */
import { useMemo, type ReactNode } from 'react';
import type { Matrix, Step } from '../../core/types';
import { Formula, Swatch } from '../../ui/components';
import { sig } from '../../core/format';
import type { StepGeometry } from './geometry';
import { modelError, stepGeometry } from './geometry';
import { texMat, texNum, texPt } from './tex';
import { FitBox } from './FitBox';
import styles from './SystemsLab.module.css';

export interface StepPanelProps {
  methodId: string;
  name: string;
  slot: number;
  geometry: StepGeometry | null;
  /** The focused run's trace: the panel sizes itself for its largest step. */
  trace: readonly Step[];
  /** The exact Jacobian (to measure a Broyden matrix against it). */
  jac: (x: number[]) => Matrix;
  /** Shown when there is no step to explain (the run stopped at 𝐱₀). */
  emptyNote: string;
}

const HALF: Record<string, string> = {
  '1': '1',
  '0.5': '1/2',
  '0.25': '1/4',
  '0.125': '1/8',
};
const texAlpha = (a: number) => HALF[String(a)] ?? texNum(a, 3);

interface StepContent {
  tex: string;
  caption: ReactNode;
  /** A plain-text proxy of the caption's length (to pick the longest caption of a run). */
  captionSize: number;
}

/** The step's formula and caption (pure: also used to size the panel for the whole run). */
function stepContent(methodId: string, g: StepGeometry, jac: (x: number[]) => Matrix): StepContent {
  const k = g.j - 1;
  const xk = `\\mathbf{x}_{${k}}`;
  const xk1 = `\\mathbf{x}_{${k + 1}}`;
  if (methodId === 'broyden' && g.secant) {
    const err = modelError(g.M, jac(g.from));
    const errTex = `\\|B_{${k}} - J(${xk})\\|_F / \\|J(${xk})\\|_F = ${texNum(err, 3)}`;
    return {
      tex:
        `\\begin{aligned} B_{${k}} &= ${texMat(g.M)} \\\\ F(${xk}) &= ${texPt(g.F)}^{\\mathsf T} \\\\ ` +
        `\\mathbf{s}_{${k}} &= -B_{${k}}^{-1} F(${xk}) = ${texPt(g.secant.s)}^{\\mathsf T} \\\\ ` +
        `\\mathbf{y}_{${k}} &= F(${xk1}) - F(${xk}) = ${texPt(g.secant.y)}^{\\mathsf T} \\end{aligned}`,
      caption: (
        <span>
          <Formula tex={`B_{${k + 1}}\\mathbf{s}_{${k}} = \\mathbf{y}_{${k}}`} /> defines the next
          model (the secant equation). Model error <Formula tex={errTex} />.
        </span>
      ),
      captionSize: 70 + errTex.length / 2,
    };
  }
  const a = g.alpha;
  const step =
    a === 1 ? `${xk} + \\mathbf{p}_{${k}}` : `${xk} + ${texAlpha(a)}\\,\\mathbf{p}_{${k}}`;
  const tex =
    `\\begin{aligned} J(${xk}) &= ${texMat(g.M)} \\\\ F(${xk}) &= ${texPt(g.F)}^{\\mathsf T} \\\\ ` +
    `\\mathbf{p}_{${k}} &= -J(${xk})^{-1} F(${xk}) = ${texPt(g.p)}^{\\mathsf T} \\\\ ` +
    `${xk1} &= ${step} = ${texPt(g.to)} \\end{aligned}`;
  if (g.trials.length > 0) {
    const rejected = g.trials.length > 1 ? g.trials.length - 1 : 0;
    const accepted = `\\alpha_{${k}} = ${texAlpha(a)},\\ \\phi = ${texNum(g.trials[g.trials.length - 1][1], 3)}`;
    return {
      tex,
      caption: (
        <span>
          Backtracking on <Formula tex="\phi = \|F\|_2^2/2" />: {g.trials.length}{' '}
          {g.trials.length === 1 ? 'trial' : 'trials'}
          {rejected > 0 ? `, ${rejected} rejected` : ''}; accepted <Formula tex={accepted} />.
        </span>
      ),
      captionSize: 50 + accepted.length / 2,
    };
  }
  return {
    tex,
    caption: (
      <span>
        The lines <Formula tex="\ell_1, \ell_2" /> where the two linear models vanish cross at{' '}
        <Formula tex={`${xk} + \\mathbf{p}_{${k}}`} />.
      </span>
    ),
    captionSize: 70,
  };
}

/** The `n` largest entries by `size`, largest first. */
function largest<T>(items: readonly T[], size: (t: T) => number, n: number): T[] {
  return [...items].sort((p, q) => size(q) - size(p)).slice(0, n);
}

export function StepPanel({
  methodId,
  name,
  slot,
  geometry: g,
  trace,
  jac,
  emptyNote,
}: StepPanelProps) {
  // Every step of the run: the panel reserves the largest formula and caption among them, so it
  // keeps one size (and one fit scale) while the playback steps through the run.
  const all = useMemo(() => {
    const out: StepContent[] = [];
    for (let j = 1; j < trace.length; j++) {
      const gj = stepGeometry(trace, j);
      if (gj) out.push(stepContent(methodId, gj, jac));
    }
    return out;
  }, [trace, methodId, jac]);
  const texSizers = useMemo(
    () =>
      largest(all, (c) => c.tex.length, 3).map((c) => <Formula key={c.tex} tex={c.tex} display />),
    [all],
  );
  const captionSizers = useMemo(() => largest(all, (c) => c.captionSize, 2), [all]);

  if (!g) {
    return (
      <div className={styles.stepPanel}>
        <p className={styles.stepEmpty}>{emptyNote}</p>
      </div>
    );
  }
  const k = g.j - 1;
  const { tex, caption } = stepContent(methodId, g, jac);
  return (
    <div className={styles.stepPanel} aria-label={`${name}, step ${k} to ${k + 1}`} role="group">
      <p className={styles.stepHead}>
        <Swatch slot={slot} size={9} />
        <span className={styles.stepK}>
          <Formula tex={`k = ${k} \\to ${k + 1}`} />
          {' · '}
          <Formula tex={`\\|\\mathbf{x}_{${k + 1}} - \\mathbf{x}_{${k}}\\|_2 =`} />{' '}
          <span className={styles.mono}>
            {sig(Math.hypot(g.to[0] - g.from[0], g.to[1] - g.from[1]), 3)}
          </span>
        </span>
      </p>
      {/* Fits width and height: the last row (the step's result) must stay visible. */}
      <FitBox
        className={styles.stepFormula}
        fitKey={`${methodId}|${trace.length}|${all.length}`}
        sizers={texSizers}
      >
        <Formula tex={tex} display />
      </FitBox>
      {/* The caption reserves the height of the longest caption of the run. */}
      <div className={styles.stepCaptionBox}>
        <p className={styles.stepCaption}>{caption}</p>
        {captionSizers.map((c, i) => (
          <p key={i} className={styles.stepCaption} aria-hidden="true" data-sizer="">
            {c.caption}
          </p>
        ))}
      </div>
    </div>
  );
}
