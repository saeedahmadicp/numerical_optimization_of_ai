/**
 * The augmented matrix of a direct method, stage by stage (the lab's main view for elimination
 * and factorizations). Between two steps of the trace the stage is played in three beats, read
 * from the fractional playhead: the row swap (rows slide, keyed by their original equation),
 * the multipliers (chips beside the rows they will change), then the update (changed entries
 * flash, eliminated entries become 0). On the right the factor that the stage records builds
 * up: L (with the permutation), Cholesky's L, the Householder vector, or Thomas's c′ and d′.
 *
 * Timing: on the interval [k, k + 1) the view rests on step k for the first part, then plays the
 * beats of step k + 1, and the update lands as the playhead reaches k + 1. The step number, the
 * method card, the table and the playback bar therefore all read k until the update is done; the
 * caption says "k = 3 → 4" while the beats run. The solution footer appears at the last step.
 *
 * Rows are numbered by position (R₁ … Rₙ, as in the captions and the row operations); a row that
 * a swap has moved carries the number of its original equation ("eq 2").
 */
import { motion } from 'motion/react';
import { useRef, type CSSProperties, type ReactNode } from 'react';
import { useElementSize } from '../../viz';
import { baseTransition } from '../../ui/motion';
import type { Matrix, Step, Vector } from '../../core/types';
import { Formula } from '../../ui/components';
import { msgText } from './messages';
import { lowerFactors, stageOf, swapped, type StageView } from './model';
import { cellText, fmt, pivotSymbol, texNum, thomasStageTex } from './texfmt';
import styles from './EliminationView.module.css';

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
/** Part of the interval [k, k + 1) spent at rest on step k before the beats of step k + 1. */
const REST = 0.35;
const subscript = (n: number) =>
  String(n)
    .split('')
    .map((d) => SUBS[Number(d)])
    .join('');

type Beat = 'swap' | 'mult' | 'apply' | 'rest';

export interface EliminationViewProps {
  method: string;
  trace: readonly Step[];
  /** Local playhead of this run (continuous). */
  t: number;
  n: number;
  slot: number;
  reduced: boolean;
  /** Announce each step to screen readers (true when the player is paused or stepped). */
  announce?: boolean;
  /** Result fields for the final summary. */
  converged: boolean;
  message: string;
  extra: Record<string, unknown>;
}

export function EliminationView(props: EliminationViewProps) {
  const { method, trace, t, n, reduced } = props;
  const wrapRef = useRef<HTMLDivElement>(null);
  const { width } = useElementSize(wrapRef);
  const last = trace.length - 1;
  const tf = Math.max(0, Math.min(t, last));
  const k = Math.floor(tf + 1e-9);
  const u = tf - k;
  const animating = !reduced && u > REST && k < last;
  const v = animating ? (u - REST) / (1 - REST) : 0;
  const base = stageOf(trace[k], n);
  const target = animating ? stageOf(trace[k + 1], n) : base;
  const beat: Beat = !animating ? 'rest' : v < 0.4 ? 'swap' : v < 0.75 ? 'mult' : 'apply';
  /** The step whose matrix and factors are drawn (the next one once its update lands). */
  const kShown = beat === 'apply' ? k + 1 : k;
  // The step before the one drawn (for "what changed").
  const before = kShown > 0 ? stageOf(trace[kShown - 1], n) : null;

  const preUpdate = beat === 'swap' || beat === 'mult';
  const matrix = preUpdate ? swapped(base.matrix, target.rowSwap) : target.matrix;
  const perm = preUpdate ? swapped(base.perm, target.rowSwap) : target.perm;
  const showChips = beat !== 'swap' && target.phase === 'eliminate' && !target.zeroPivot;
  const reference = before ? swapped(before.matrix, target.rowSwap) : null;
  const changed = (i: number, j: number) =>
    !preUpdate &&
    reference !== null &&
    reference[i]?.[j] !== undefined &&
    reference[i][j] !== matrix[i][j];

  const augmented = (matrix[0]?.length ?? 0) > n;
  const cols = matrix[0]?.length ?? n;
  const c = target.pivot?.[1] ?? -1;
  const dense = n > 6;
  const size = dense ? 's' : n > 4 ? 'm' : 'l';
  // Wide enough for the longest entry this run ever shows (no jumps between stages).
  let chars = 1;
  for (const s of trace)
    for (const row of (s.info.matrix as Matrix | undefined) ?? [])
      for (const v of row) chars = Math.max(chars, cellText(v).length);
  const glyph = { s: 7.1, m: 8.4, l: 9.6 }[size];
  const natural = Math.max({ s: 40, m: 56, l: 76 }[size], Math.ceil(chars * glyph + 20));
  // Fit the stage width: first drop the multiplier column (the caption states the operation),
  // then shrink the cells and their type together (phones).
  const avail = Math.max(0, width - 4);
  let chipW = dense ? 66 : 92;
  let cell = natural;
  let fontScale = 1;
  if (avail > 0 && 30 + chipW + cols * cell > avail) {
    chipW = 0;
    if (30 + cols * cell > avail) {
      cell = Math.max(30, Math.floor((avail - 30) / cols));
      fontScale = Math.max(0.62, cell / natural);
    }
  }
  const tmpl = `30px ${chipW}px repeat(${cols}, ${cell}px)`;

  const Ls = lowerFactors(method, trace, n);
  // Before the update beat the factor is still the current one.
  const L = Ls[kShown] ?? null;
  const Lprev = beat === 'apply' && kShown > 0 ? (Ls[kShown - 1] ?? null) : null;

  /** "− m·R_c" with the sign folded in ("+ 0.5·R₁" for m = −0.5); rows by position. */
  const op = (m: number, row: string) => `${m < 0 ? '+' : '−'} ${fmt(Math.abs(m))}·R${row}`;
  const chip = (i: number): ReactNode => {
    if (!showChips || c < 0) return null;
    const m = target.multipliers;
    const pr = subscript(c + 1);
    if (method === 'gauss_jordan') {
      if (i === c) return <span className={styles.chip}>÷ {fmt(target.pivotValue)}</span>;
      if (!m || m[i] === 0) return null;
      return <span className={styles.chip}>{op(m[i], pr)}</span>;
    }
    if (method === 'thomas')
      return i === c ? <span className={styles.chip}>÷ {fmt(target.pivotValue)}</span> : null;
    if (method === 'qr_householder')
      return i === c ? <span className={styles.chip}>H{subscript(c + 1)}</span> : null;
    if (method === 'cholesky') return null;
    if (!m || i <= c || m[i] === 0) return null;
    return (
      <span
        className={styles.chip}
        title={`R${i + 1} ← R${i + 1} − (${fmt(m[i])})·R${c + 1}: subtract ${fmt(m[i])} times the pivot row`}
      >
        {op(m[i], pr)}
      </span>
    );
  };

  const band = (i: number, j: number) => {
    if (c < 0 || target.phase !== 'eliminate') return false;
    if (method === 'qr_householder') return i >= c && j >= c;
    if (method === 'cholesky')
      return preUpdate ? i >= c && j >= c && j < n : i > c && j > c && j < n;
    if (method === 'gauss_jordan') return j === c;
    if (method === 'thomas') return i === c;
    return i > c && j >= c;
  };

  return (
    <div className={styles.wrap}>
      <Caption {...props} stage={target} k={k} next={animating ? k + 1 : null} beat={beat} />
      <span className="visually-hidden" aria-live="polite" aria-atomic="true">
        {props.announce ? announcement(props.method, base, k, last, n) : ''}
      </span>
      <div className={styles.body} ref={wrapRef}>
        <figure className={styles.fig}>
          <div
            className={styles.grid}
            role="table"
            aria-label={`${augmented ? 'Augmented matrix' : 'Working matrix'} at step ${kShown}; rows R1 to R${matrix.length} by position`}
            data-size={size}
            style={{ ['--tmpl' as string]: tmpl, ['--fit' as string]: fontScale } as CSSProperties}
          >
            <div className={styles.row} role="row" style={{ gridTemplateColumns: tmpl }}>
              <span role="columnheader">
                <span className="visually-hidden">Equation</span>
              </span>
              <span role="columnheader">
                <span className="visually-hidden">Row operation</span>
              </span>
              {Array.from({ length: cols }, (_, j) => (
                <span
                  key={j}
                  role="columnheader"
                  className={`${styles.colHead} ${augmented && j === n ? styles.sepCol : ''}`}
                >
                  {j < n ? (
                    <>
                      <i>x</i>
                      {subscript(j + 1)}
                    </>
                  ) : (
                    <i>b</i>
                  )}
                </span>
              ))}
            </div>
            {matrix.map((row, i) => (
              <motion.div
                key={perm[i]}
                layout={reduced ? false : 'position'}
                transition={baseTransition}
                className={styles.row}
                role="row"
                data-active={target.phase === 'eliminate' && i === c ? 'true' : undefined}
                style={{ gridTemplateColumns: tmpl }}
              >
                <span role="rowheader" className={styles.rowHead}>
                  <span>R{subscript(i + 1)}</span>
                  {perm[i] !== i && (
                    <span
                      className={styles.eqTag}
                      title={`This row is equation ${perm[i] + 1} of the original system`}
                    >
                      eq {perm[i] + 1}
                    </span>
                  )}
                </span>
                <span className={styles.chipCell}>{chipW > 0 ? chip(i) : null}</span>
                {row.map((v, j) => (
                  <span
                    key={`${j}-${kShown}-${beat === 'apply' || beat === 'rest' ? 1 : 0}`}
                    role="cell"
                    className={`${styles.cell} ${augmented && j === n ? styles.sepCol : ''}`}
                    data-zero={v === 0 || undefined}
                    data-band={band(i, j) || undefined}
                    data-pivot={
                      target.pivot &&
                      target.pivot[0] === i &&
                      target.pivot[1] === j &&
                      !(method === 'cholesky' && !preUpdate && !target.zeroPivot)
                        ? target.zeroPivot
                          ? 'bad'
                          : 'true'
                        : undefined
                    }
                    data-changed={changed(i, j) || undefined}
                    data-upper={!augmented || j < n ? (j >= i ? 'u' : undefined) : undefined}
                  >
                    {cellText(v)}
                  </span>
                ))}
              </motion.div>
            ))}
          </div>
          <figcaption className={styles.figcap}>
            {matrixCaption(method, target, augmented)}
          </figcaption>
        </figure>

        <Factors
          method={method}
          step={trace[kShown]}
          L={L}
          Lprev={Lprev}
          c={c}
          perm={perm}
          dense={dense}
        />
      </div>
      <Solution {...props} step={trace[k]} atEnd={k === last} />
    </div>
  );
}

function matrixCaption(method: string, s: StageView, augmented: boolean): ReactNode {
  if (s.phase === 'solve')
    return method === 'cholesky' ? (
      <>
        <Formula tex="L^{\mathsf{T}}" /> (upper triangular)
      </>
    ) : (
      <>
        <Formula tex="U" /> = the finished working matrix
      </>
    );
  if (method === 'cholesky')
    return (
      <>
        <Formula tex="W" />: the active Schur complement; finished rows and columns are 0
      </>
    );
  if (method === 'lu_decomposition')
    return (
      <>
        <Formula tex="W" />: rows of <Formula tex="U" /> above the pivot, Schur complement below
      </>
    );
  if (method === 'qr_householder')
    return (
      <>
        <Formula tex="[\,R \mid Q^{\mathsf{T}}\mathbf{b}\,]" /> in progress
      </>
    );
  if (method === 'gauss_jordan')
    return (
      <>
        <Formula tex="[\,A \mid \mathbf{b}\,] \to [\,I \mid \mathbf{x}\,]" />
      </>
    );
  if (method === 'thomas')
    return (
      <>
        Row <Formula tex="i" /> becomes <Formula tex="(0,\dots,1,\,c'_i,\dots \mid d'_i)" />
      </>
    );
  return augmented ? (
    <>
      <Formula tex="[\,A \mid \mathbf{b}\,] \to [\,U \mid \tilde{\mathbf{b}}\,]" />
    </>
  ) : null;
}

// ── Caption: what this stage does, with the numbers filled in ─────────────────────────

function Caption({
  method,
  n,
  stage,
  k,
  next,
  beat,
}: EliminationViewProps & { stage: StageView; k: number; next: number | null; beat: Beat }) {
  const c = stage.pivot?.[1] ?? -1;
  const ci = c + 1;
  let lead: ReactNode;
  let tex: string | null;
  if (stage.phase === 'start') {
    lead =
      method === 'lu_decomposition' || method === 'cholesky'
        ? 'Start: A is factored in place'
        : 'Start: the augmented matrix';
    tex =
      method === 'lu_decomposition' || method === 'cholesky'
        ? 'W = A'
        : '[\\,A \\mid \\mathbf{b}\\,]';
  } else if (stage.phase === 'back_substitution') {
    lead = method === 'thomas' ? 'Back sweep' : 'Back substitution, from the last row up';
    tex =
      method === 'thomas'
        ? "x_{n} = d'_{n},\\quad x_i = d'_i - c'_i\\,x_{i+1}"
        : 'x_i = \\Big(\\tilde b_i - \\sum_{j>i} u_{ij}x_j\\Big)\\Big/u_{ii}';
  } else if (stage.phase === 'solve') {
    lead = 'Two triangular solves';
    tex =
      method === 'cholesky'
        ? 'L\\mathbf{y} = \\mathbf{b},\\quad L^{\\mathsf{T}}\\mathbf{x} = \\mathbf{y}'
        : 'L\\mathbf{y} = P\\mathbf{b},\\quad U\\mathbf{x} = \\mathbf{y}';
  } else if (stage.zeroPivot) {
    lead = (
      <span className={styles.bad}>
        Stage {ci} of {n}: the pivot is zero to working precision, so the method stops
      </span>
    );
    tex = `|${pivotSymbol(method, c)}| = ${texNum(Math.abs(stage.pivotValue ?? 0), 3)} \\le \\tau`;
  } else {
    const p = texNum(stage.pivotValue);
    lead = (
      <>
        Stage {ci} of {n}
        {stage.rowSwap && beat !== 'rest' && beat !== 'apply' ? ' · swap' : ''}
      </>
    );
    const swap = stage.rowSwap
      ? `R_{${stage.rowSwap[0] + 1}} \\leftrightarrow R_{${stage.rowSwap[1] + 1}},\\quad `
      : '';
    switch (method) {
      case 'cholesky':
        tex = `l_{${ci}${ci}} = \\sqrt{w_{${ci}${ci}}} = \\sqrt{${p}},\\quad W \\leftarrow W - \\boldsymbol{\\ell}_{${ci}}\\boldsymbol{\\ell}_{${ci}}^{\\mathsf{T}}`;
        break;
      case 'qr_householder':
        tex = `H_{${ci}} = I - 2\\mathbf{v}\\mathbf{v}^{\\mathsf{T}},\\quad r_{${ci}${ci}} = ${p}`;
        break;
      case 'thomas':
        tex = thomasStageTex(ci, n, p);
        break;
      case 'gauss_jordan':
        tex = `${swap}R_{${ci}} \\leftarrow R_{${ci}}/${wrapNeg(p)},\\quad R_i \\leftarrow R_i - a_{i${ci}}R_{${ci}}\\ (i \\ne ${ci})`;
        break;
      case 'lu_decomposition':
        tex = `${swap}\\ell_{i${ci}} = w_{i${ci}}/${wrapNeg(p)}\\ (i > ${ci})`;
        break;
      default:
        tex = `${swap}a_{${ci}${ci}} = ${p},\\quad R_i \\leftarrow R_i - \\tfrac{a_{i${ci}}}{a_{${ci}${ci}}}R_{${ci}}\\ (i > ${ci})`;
    }
  }
  return (
    <div className={styles.caption}>
      <span className={styles.stepNo}>
        k = {k}
        {next !== null && <span className={styles.stepNext}> → {next}</span>}
      </span>
      <span className={styles.lead}>{lead}</span>
      {tex && (
        <span className={styles.tex}>
          <Formula tex={tex} />
        </span>
      )}
    </div>
  );
}

const wrapNeg = (s: string) => (s.startsWith('-') ? `(${s})` : s);

/** One plain sentence per step for screen readers (no TeX, no beats). */
function announcement(method: string, s: StageView, k: number, last: number, n: number): string {
  const head = `Step ${k} of ${last}`;
  if (s.phase === 'start') return `${head}: the starting matrix.`;
  if (s.phase === 'back_substitution')
    return `${head}: ${method === 'thomas' ? 'back sweep' : 'back substitution'}.`;
  if (s.phase === 'solve') return `${head}: two triangular solves.`;
  const c = (s.pivot?.[1] ?? 0) + 1;
  if (s.zeroPivot) return `${head}: stage ${c} of ${n}, zero pivot; the method stops.`;
  const swap = s.rowSwap ? `, rows ${s.rowSwap[0] + 1} and ${s.rowSwap[1] + 1} swapped` : '';
  return `${head}: stage ${c} of ${n}, pivot ${fmt(s.pivotValue)}${swap}.`;
}

// ── Factors panel ─────────────────────────────────────────────────────────────────────

function SmallMatrix({
  M,
  prev,
  label,
  highlightCol,
  dense,
  lower = false,
}: {
  M: Matrix;
  prev?: Matrix | null;
  label: ReactNode;
  highlightCol?: number;
  dense: boolean;
  /** Leave the (zero) upper triangle blank, so the triangular shape reads at a glance. */
  lower?: boolean;
}) {
  const m = M[0]?.length ?? 0;
  let chars = 1;
  for (const row of M) for (const v of row) chars = Math.max(chars, cellText(v).length);
  const w = Math.max(dense ? 34 : 44, Math.ceil(chars * 6.9 + 14));
  return (
    <figure className={styles.factor}>
      <div className={styles.factorLabel}>{label}</div>
      <div
        className={styles.small}
        role="table"
        aria-label="Factor"
        style={{ gridTemplateColumns: `repeat(${m}, ${w}px)` }}
      >
        {M.map((row, i) => (
          <div key={i} role="row" style={{ display: 'contents' }}>
            {row.map((v, j) => {
              const was = prev?.[i]?.[j];
              const isNew =
                prev !== undefined &&
                prev !== null &&
                !Object.is(was, v) &&
                !(Number.isNaN(was) && Number.isNaN(v));
              return (
                <span
                  key={`${j}-${isNew ? 'n' : 'o'}`}
                  role="cell"
                  className={styles.smallCell}
                  data-unknown={Number.isNaN(v) || undefined}
                  data-zero={v === 0 || undefined}
                  data-new={isNew || undefined}
                  data-col={highlightCol === j || undefined}
                >
                  {lower && j > i ? '' : cellText(v)}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </figure>
  );
}

function VectorCol({
  v,
  label,
  prev,
  dense,
}: {
  v: Vector;
  label: ReactNode;
  prev?: Vector | null;
  dense: boolean;
}) {
  return (
    <SmallMatrix
      M={v.map((x) => [x])}
      prev={prev ? prev.map((x) => [x]) : null}
      label={label}
      dense={dense}
    />
  );
}

function Factors({
  method,
  step,
  L,
  Lprev,
  c,
  perm,
  dense,
}: {
  method: string;
  step: Step;
  L: Matrix | null;
  Lprev: Matrix | null;
  c: number;
  perm: number[];
  dense: boolean;
}) {
  const info = step.info;
  const panels: ReactNode[] = [];
  if (L)
    panels.push(
      <SmallMatrix
        key="L"
        M={L}
        prev={Lprev}
        dense={dense}
        lower
        highlightCol={c}
        label={
          method === 'cholesky' ? (
            <>
              <Formula tex="L" /> with <Formula tex="A = LL^{\mathsf{T}}" />
            </>
          ) : (
            <>
              <Formula tex="L" />, unit lower triangular
            </>
          )
        }
      />,
    );
  if (method === 'qr_householder' && Array.isArray(info.householder_vector))
    panels.push(
      <VectorCol
        key="v"
        v={info.householder_vector as Vector}
        dense={dense}
        label={
          <>
            <Formula tex="\mathbf{v}" /> of{' '}
            <Formula tex="H = I - 2\mathbf{v}\mathbf{v}^{\mathsf{T}}" />
          </>
        }
      />,
    );
  if (method === 'thomas' && Array.isArray(info.c_prime)) {
    panels.push(
      <VectorCol
        key="c"
        v={(info.c_prime as Vector).map((v, i) => (i <= c || c < 0 ? v : NaN))}
        dense={dense}
        label={<Formula tex="\mathbf{c}'" />}
      />,
      <VectorCol
        key="d"
        v={(info.d_prime as Vector).map((v, i) => (i <= c || c < 0 ? v : NaN))}
        dense={dense}
        label={<Formula tex="\mathbf{d}'" />}
      />,
    );
  }
  if (Array.isArray(info.y))
    panels.push(
      <VectorCol key="y" v={info.y as Vector} dense={dense} label={<Formula tex="\mathbf{y}" />} />,
    );
  const pivoting =
    method === 'gaussian_elimination_pivoting' ||
    method === 'lu_decomposition' ||
    method === 'gauss_jordan';
  return (
    <div className={styles.factors}>
      {panels}
      {pivoting && (
        <p className={styles.perm}>
          <Formula tex="P" />: rows in the order{' '}
          <span className={styles.mono}>({perm.map((p) => p + 1).join(', ')})</span>
        </p>
      )}
    </div>
  );
}

// ── Solution footer ───────────────────────────────────────────────────────────────────

function Solution({
  step,
  atEnd,
  converged,
  message,
  extra,
}: EliminationViewProps & { step: Step; atEnd: boolean }) {
  const x = Array.isArray(step.x) ? (step.x as number[]) : null;
  if (!atEnd && !x) return <div className={styles.solutionEmpty} />;
  const eta = extra.backward_error as number | undefined;
  const bound = extra.backward_error_bound as number | undefined;
  const rho = extra.growth_factor as number | null | undefined;
  const kappa = extra.cond_estimate as number | undefined;
  return (
    <div className={styles.solution} data-tone={!atEnd ? undefined : converged ? 'good' : 'bad'}>
      {x ? (
        <div className={styles.solRow}>
          <Formula tex={`\\mathbf{x} = (${x.map((v) => texNum(v, 6)).join(',\\ ')})`} />
          {step.fun !== null && (
            <span className={styles.solMeta}>
              <Formula tex="\|\mathbf{b} - A\mathbf{x}\|_2" /> = {fmt(step.fun)}
            </span>
          )}
        </div>
      ) : (
        <p className={styles.solFail}>{msgText(message)}</p>
      )}
      {atEnd && (
        <div className={styles.solStats}>
          {eta !== undefined && (
            <span>
              backward error <Formula tex="\eta_\infty" /> = {fmt(eta)}{' '}
              {eta <= (bound ?? 0) ? '≤' : '>'} {fmt(bound)}
            </span>
          )}
          {rho !== undefined && rho !== null && (
            <span>
              growth <Formula tex="\rho" /> = {fmt(rho)}
            </span>
          )}
          {kappa !== undefined && Number.isFinite(kappa) && (
            <span>
              <Formula tex="\hat\kappa_1(A)" /> = {fmt(kappa)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
