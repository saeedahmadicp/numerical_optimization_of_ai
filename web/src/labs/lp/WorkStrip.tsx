/**
 * What the focused method is working on at the current step: the simplex tableau with the pivot
 * row and column, the ratio test as an extra column (a row for the dual simplex) and the
 * winning ratio marked; the branch-and-bound tree as it stands; or the interior-point measures.
 * A sentence in the brand voice says what the step does and why.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { LinearProgram, Step } from '../../core/types';
import { Formula } from '../../ui/components/Formula';
import { MatrixView } from '../../viz/MatrixView';
import { matrixColumnChars } from '../../viz/matrixColumns';
import { TreeView, type TreeNode, type TreeNodeStatus } from '../../viz/TreeView';
import { int } from '../../core/format';
import { objectiveRows, type TableauInfo } from './geometry';
import { cutTex, fmt, texNum, texSci, texVec, varTex } from './explain';
import { PDHG } from './pdhgView';
import styles from './LPLab.module.css';

export interface WorkRun {
  id: string;
  name: string;
  slot: number;
  trace: Step[];
  params: Record<string, unknown>;
  converged: boolean;
  status: string;
}

const F = ({ tex }: { tex: string }) => <Formula tex={tex} />;

// ── Tableau ───────────────────────────────────────────────────────────────────────────────

function tableauMatrix(info: TableauInfo, dual: boolean) {
  const nObj = objectiveRows(info);
  const N = info.col_labels.length - 1;
  const T = info.tableau;
  const ratios = info.ratio_test ?? [];
  if (dual) {
    const extra = Array.from({ length: N + 1 }, (_, j) => (j < N ? (ratios[j] ?? NaN) : NaN));
    return { matrix: [...T, extra], nObj, N, extraRow: true };
  }
  const matrix = T.map((row, r) => [...row, r >= nObj ? (ratios[r - nObj] ?? NaN) : NaN]);
  return { matrix, nObj, N, extraRow: false };
}

function TableauPanel({ run, k, compact }: { run: WorkRun; k: number; compact: boolean }) {
  const step = run.trace[k];
  const info = step.info as unknown as TableauInfo & Record<string, unknown>;
  const dual = run.id === 'dual_simplex';
  const { matrix, nObj, N } = tableauMatrix(info, dual);
  const prevStep = run.trace[k - 1];
  let previous: number[][] | null = null;
  if (prevStep) {
    const p = tableauMatrix(prevStep.info as unknown as TableauInfo, dual).matrix;
    if (p.length === matrix.length && p[0].length === matrix[0].length) previous = p;
  }
  const q = info.entering;
  const r = info.pivot_row;
  const rowLabels = [
    ...info.row_labels.map((l) => <F key={l} tex={varTex(l)} />),
    ...(dual ? [<F key="ratio" tex="\tfrac{\bar c_j}{|\bar a_{rj}|}" />] : []),
  ];
  const colLabels = [
    ...info.col_labels.map((l) => <F key={l} tex={varTex(l)} />),
    ...(dual ? [] : [<F key="theta" tex="\theta_i" />]),
  ];
  const digits = compact ? 4 : 5;
  // One width per column for the whole run (the widest value of every step's tableau, ratio
  // column included): b̄ and θ never move when a longer number appears mid-run.
  const colCh = useMemo(
    () =>
      matrixColumnChars(
        run.trace
          .filter((s) => Array.isArray(s.info.tableau))
          .map((s) => tableauMatrix(s.info as unknown as TableauInfo, dual).matrix),
        digits,
      ),
    [run.trace, dual, digits],
  );
  let win: [number, number] | null = null;
  if (!dual && r !== null && r !== undefined) win = [r, N + 1];
  if (dual && q !== null && q !== undefined) win = [matrix.length - 1, q];
  return (
    <ScrollRegion label={`Tableau of ${run.name}, scrolls sideways`}>
      <MatrixView
        matrix={matrix}
        previous={previous}
        rowLabels={rowLabels}
        colLabels={colLabels}
        pivot={
          q !== null && q !== undefined && r !== null && r !== undefined ? { row: r, col: q } : null
        }
        rows={r !== null && r !== undefined ? [r] : []}
        cols={q !== null && q !== undefined ? [q] : []}
        cells={win ? [win] : []}
        separatorBefore={N}
        digits={digits}
        colCh={colCh}
        ariaLabel={`Tableau of ${run.name} at step ${k}: ${nObj} objective row${nObj > 1 ? 's' : ''}, ${matrix.length - nObj - (dual ? 1 : 0)} constraint rows, ${N} columns`}
      />
    </ScrollRegion>
  );
}

/**
 * The tableau's scroller: a keyboard-focusable region (arrow keys scroll it) with a fade on the
 * right edge while columns remain off screen.
 */
function ScrollRegion({ label, children }: { label: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [more, setMore] = useState(false);
  const update = useCallback(() => {
    const el = ref.current;
    if (el) setMore(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    if (el.firstElementChild) ro.observe(el.firstElementChild);
    return () => ro.disconnect();
  }, [update]);
  return (
    <div
      ref={ref}
      className={styles.tableauScroll}
      tabIndex={0}
      role="region"
      aria-label={label}
      data-more={more || undefined}
      onScroll={update}
    >
      {children}
    </div>
  );
}

/** Traces up to this length lay every step's sentence out in one grid cell (see NoteStack). */
const STACK_MAX = 120;

/**
 * Every step's sentence in one grid cell; only step k's is visible, so the strip keeps the height
 * of its longest sentence during playback (no layout shift). The sentences are built once per
 * run, so playback re-renders only the visibility.
 */
function NoteStack({ n, k, render }: { n: number; k: number; render: (i: number) => ReactNode }) {
  const all = useMemo(
    () => (n <= STACK_MAX ? Array.from({ length: n }, (_, i) => render(i)) : null),
    [n, render],
  );
  if (!all) return <span className={styles.noteFallback}>{render(k)}</span>;
  return (
    <span className={styles.noteStack}>
      {all.map((node, i) => (
        <span key={i} aria-hidden={i === k ? undefined : true}>
          {node}
        </span>
      ))}
    </span>
  );
}

function ruleWords(run: WorkRun): string {
  const rule = String(run.params.pivot_rule ?? 'dantzig');
  if (rule === 'bland') return 'the smallest index with a negative reduced cost (Bland)';
  if (rule === 'steepest_edge') return 'the steepest edge, the most negative c̄ⱼ/‖ηⱼ‖';
  return 'the most negative reduced cost (Dantzig)';
}

/** One or two sentences: what this tableau step does. */
function TableauNote({ run, k }: { run: WorkRun; k: number }): ReactNode {
  const step = run.trace[k];
  const info = step.info as unknown as TableauInfo & Record<string, unknown>;
  const q = info.entering;
  const lab = (j: number | null | undefined) =>
    j === null || j === undefined ? '' : varTex(info.col_labels[j]);
  if (run.id === 'gomory_cuts') return <GomoryNote run={run} k={k} />;
  if (run.id === 'dual_simplex') {
    if (q === null && info.leaving !== null && info.leaving !== undefined)
      return (
        <>
          <F tex={lab(info.leaving)} /> is negative and its row has no negative entry: no point with{' '}
          <F tex="\mathbf z \ge \mathbf 0" /> satisfies it. The LP is infeasible.
        </>
      );
    if (q === null || q === undefined)
      return info.cycling ? (
        <>This basis occurred before: the dual simplex cycles.</>
      ) : (
        <>
          Every basic value is <F tex="\ge 0" /> and every reduced cost is <F tex="\ge 0" />: the
          basis is primal and dual feasible, so{' '}
          <F tex={`\\mathbf x = ${texVec(step.x as number[])}`} /> is optimal.
        </>
      );
    const N = info.col_labels.length - 1;
    const rr = info.pivot_row as number;
    return (
      <>
        <F tex={lab(info.leaving)} /> leaves: its value <F tex={texNum(info.tableau[rr][N])} /> is
        negative. <F tex={lab(q)} /> enters: the smallest ratio <F tex="\bar c_j/|\bar a_{rj}|" />{' '}
        keeps every reduced cost <F tex="\ge 0" />.
      </>
    );
  }
  if (q === null || q === undefined) {
    if (info.cycling)
      return (
        <>
          This basis occurred before: with {ruleWords(run)} the method would loop for ever. Bland’s
          rule cannot cycle.
        </>
      );
    if (info.phase === 1 || (run.status === 'infeasible' && k === run.trace.length - 1))
      return run.id === 'big_m' ? (
        <>
          Every reduced cost is <F tex="\ge 0" />, but the optimum keeps an artificial variable
          positive{' '}
          <span className={styles.nowrap}>
            (<F tex={`M\\text{-part} = ${texNum(info.infeasibility as number)}`} />
            ):
          </span>{' '}
          no point satisfies the constraints.
        </>
      ) : (
        <>
          Phase 1 ends with{' '}
          <F tex={`\\textstyle\\sum_i a_i/\\rho_i = ${texNum(info.infeasibility as number)} > 0`} />
          : the artificial variables cannot all reach zero, so no feasible point exists.
        </>
      );
    return (
      <>
        Every reduced cost is <F tex="\bar c_j \ge 0" />: no edge improves the objective, so this
        vertex is optimal with <F tex={`\\mathbf c^{\\top}\\mathbf x = ${texNum(step.fun)}`} />.
      </>
    );
  }
  if (info.drive_out)
    return (
      <>
        The artificial <F tex={lab(info.leaving)} /> is basic at level 0 after phase 1. A degenerate
        pivot (<F tex="\theta = 0" />) replaces it by <F tex={lab(q)} />.
      </>
    );
  if (info.leaving === null || info.leaving === undefined)
    return (
      <>
        <F tex={lab(q)} /> enters, but no entry of its column is positive: the edge never meets a
        constraint, and the objective improves without limit along it.
      </>
    );
  const ratios = (info.ratio_test ?? []).filter((v): v is number => v !== null);
  const theta = Math.min(...ratios);
  return (
    <>
      {info.phase === 1 && <span className={styles.phase}>Phase 1 · </span>}
      <F tex={lab(q)} /> enters: {ruleWords(run)}. <F tex={lab(info.leaving)} /> leaves: it reaches
      zero first, at <F tex={`\\theta = ${texNum(theta)}`} />
      {theta <= 1e-12 ? <> — a degenerate pivot: the basis changes, the vertex does not.</> : '.'}
    </>
  );
}

function GomoryNote({ run, k }: { run: WorkRun; k: number }) {
  const step = run.trace[k];
  const cut = step.info.cut as {
    coef: number[];
    rhs: number;
    source_var: string;
    f0: number;
  } | null;
  const nCuts = (step.info.cuts as unknown[]).length;
  if (!cut)
    return (
      <>
        The LP relaxation, solved exactly in rational arithmetic:{' '}
        <F tex={`\\mathbf x = ${texVec(step.x as number[])}`} />.{' '}
        {run.trace.length > 1
          ? 'A basic value is fractional, so a cut follows.'
          : 'Every basic value is an integer.'}
      </>
    );
  return (
    <>
      Cut {nCuts} from the row of <F tex={varTex(cut.source_var)} /> (fractional part{' '}
      <F tex={texNum(cut.f0, 3)} />
      ): <F tex={cutTex(cut)} />. It holds at every integer point and excludes the last vertex;{' '}
      {int(step.info.dual_pivots as number)} dual simplex pivot
      {step.info.dual_pivots === 1 ? '' : 's'} reoptimize.
    </>
  );
}

// ── Branch and bound ──────────────────────────────────────────────────────────────────────

interface BBNode {
  id: number;
  parent: number | null;
  branch: [number, string, number] | null;
  lp_value: number | null;
  lp_x: number[] | null;
  status: string;
}

const STATUS: Record<string, TreeNodeStatus> = {
  open: 'open',
  branched: 'branched',
  integer: 'incumbent',
  pruned_bound: 'pruned',
  infeasible: 'infeasible',
  unbounded: 'infeasible',
};

function TreePanel({ run, k }: { run: WorkRun; k: number }) {
  const step = run.trace[k];
  const tree = step.info.tree as BBNode[];
  const inc = step.info.incumbent as number[] | null;
  const last = k === run.trace.length - 1;
  const nodes: TreeNode[] = tree.map((n) => {
    let status = STATUS[n.status] ?? 'open';
    const isInc =
      inc &&
      n.lp_x &&
      n.status === 'integer' &&
      n.lp_x.every((v, j) => Math.abs(v - inc[j]) < 1e-6);
    if (status === 'incumbent' && !isInc) status = 'pruned';
    if (status === 'incumbent' && last && run.converged) status = 'optimal';
    const edge = n.branch ? (
      <F
        tex={`x_{${n.branch[0] + 1}} ${n.branch[1] === '<=' ? '\\le' : '\\ge'} ${texNum(n.branch[2])}`}
      />
    ) : undefined;
    return {
      id: String(n.id),
      parent: n.parent === null ? null : String(n.parent),
      label:
        n.lp_value === null
          ? n.status === 'open'
            ? 'not solved'
            : n.status === 'infeasible'
              ? 'LP infeasible'
              : 'not solved'
          : `z = ${fmt(n.lp_value, 5)}`,
      detail: n.lp_x ? `(${n.lp_x.map((v) => fmt(v, 3)).join(', ')})` : undefined,
      edge,
      status,
      description: `Node ${n.id}${n.branch ? `, x${n.branch[0] + 1} ${n.branch[1] === '<=' ? '≤' : '≥'} ${n.branch[2]}` : ', root'}: ${n.status.replace('_', ' ')}${n.lp_value !== null ? `, LP value ${n.lp_value}` : ''}.`,
    };
  });
  return (
    <TreeView
      nodes={nodes}
      current={String(step.info.node)}
      ariaLabel={`Branch-and-bound tree at step ${k}: ${tree.length} nodes`}
      nodeWidth={150}
      nodeHeight={46}
      className={styles.treeScroll}
    />
  );
}

function TreeNote({ run, k, lp }: { run: WorkRun; k: number; lp: LinearProgram }) {
  const step = run.trace[k];
  const info = step.info as Record<string, unknown>;
  const tree = info.tree as BBNode[];
  const node = tree[info.node as number];
  const inc = info.incumbent_value as number | null;
  const bound = info.best_bound as number | null;
  const better = lp.sense === 'max' ? 'larger' : 'smaller';
  let what: ReactNode;
  if (node.status === 'branched') {
    const j = info.branch_var as number;
    const v = (node.lp_x as number[])[j];
    what = (
      <>
        The LP optimum has <F tex={`x_{${j + 1}} = ${texNum(v)}`} />, the most fractional variable:
        the box splits into <F tex={`x_{${j + 1}} \\le ${Math.floor(v)}`} /> and{' '}
        <F tex={`x_{${j + 1}} \\ge ${Math.ceil(v)}`} />, which both exclude it.
      </>
    );
  } else if (node.status === 'integer') what = <>The LP optimum is integral: a new incumbent.</>;
  else if (node.status === 'pruned_bound')
    what = (
      <>Its bound is not {better} than the incumbent’s value, so the node is pruned unexplored.</>
    );
  else if (node.status === 'infeasible') what = <>Its LP is infeasible: pruned.</>;
  else what = <>Its LP relaxation is unbounded.</>;
  return (
    <>
      Node {node.id}. {what}{' '}
      {inc !== null && bound !== null && (
        <span className={styles.muted}>
          Incumbent <F tex={texNum(inc)} />, best bound <F tex={texNum(bound)} />.
        </span>
      )}
    </>
  );
}

// ── Interior point ────────────────────────────────────────────────────────────────────────

function InteriorPanel({ run, k }: { run: WorkRun; k: number }) {
  const step = run.trace[k];
  const i = step.info as Record<string, number | null>;
  const cells: [string, number | null | undefined][] =
    run.id === 'primal_dual_ipm'
      ? [
          ['\\mu_k', i.mu],
          ['\\sigma', i.sigma],
          ['\\alpha^{\\text{aff}}_{\\text{pri}}', i.alpha_aff_primal],
          ['\\alpha_{\\text{pri}}', i.alpha_primal],
          ['\\alpha_{\\text{dual}}', i.alpha_dual],
          ['\\|\\mathbf r_b\\|', i.primal_residual],
          ['\\|\\mathbf r_c\\|', i.dual_residual],
        ]
      : [
          ['\\alpha_k', i.alpha],
          ['t_k', i.artificial],
          ['\\mathbf z^{\\top}\\mathbf r', i.gap],
          ['\\|\\mathbf r_b\\|', i.primal_residual],
          ['\\|\\min(\\mathbf r, 0)\\|', i.dual_residual],
        ];
  return (
    <dl className={styles.stats}>
      {cells.map(([tex, v]) => (
        <div key={tex}>
          <dt>
            <F tex={tex} />
          </dt>
          <dd>
            <F tex={texNum(v ?? null, 3)} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

function InteriorNote({ run, k }: { run: WorkRun; k: number }) {
  if (run.id === 'primal_dual_ipm')
    return k === 0 ? (
      <>Mehrotra’s start: a point with z, s &gt; 0 that need not satisfy the constraints.</>
    ) : (
      <>
        The predictor (dashed) goes as far as a pure Newton step can; the centering{' '}
        <F tex="\sigma = (\mu_{\text{aff}}/\mu)^3" /> pulls the corrector back toward the central
        path.
      </>
    );
  const t = Number(run.trace[k].info.artificial ?? 0);
  const long = String(run.params.variant ?? 'long') === 'long';
  const stepWords = long ? (
    <>
      the long step then goes a fraction <F tex="\beta" /> of the way to the nearest{' '}
      <F tex="z_j = 0" />.
    </>
  ) : (
    <>
      the short step goes to <F tex="\beta" /> times the ellipsoid’s boundary.
    </>
  );
  if (k === 0)
    return (
      <>
        Interior start <F tex="\mathbf z = \mathbf e" /> with one artificial variable <F tex="t" />{' '}
        that absorbs the infeasibility. The ellipse is the method’s metric{' '}
        <F tex="\|Z^{-1}\mathbf d\| \le 1" /> on{' '}
        <span className={styles.nowrap}>
          <F tex="\mathbf z = (\mathbf x, \mathbf s, t)" />,
        </span>{' '}
        projected onto the plane: while <F tex="t > 0" /> it can reach outside the polygon.
      </>
    );
  return t > 1e-6 ? (
    <>
      The ellipse is the method’s Dikin ellipsoid{' '}
      <F tex="\{\mathbf d : \|Z^{-1}\mathbf d\| \le 1,\ [A\ \ \mathbf r_0]\,\mathbf d = \mathbf 0\}" />{' '}
      over{' '}
      <span className={styles.nowrap}>
        <F tex="\mathbf z = (\mathbf x, \mathbf s, t)" />,
      </span>{' '}
      projected onto the plane. While <F tex={`t_k = ${texNum(t, 3)} > 0`} /> the artificial moves
      too, so it is not yet the polygon’s Dikin ellipse; {stepWords}
    </>
  ) : (
    <>
      With <F tex="t_k \approx 0" /> the ellipse is the polygon’s Dikin ellipse, the unit ball of
      the rescaled metric in which the step is steepest descent; {stepWords}
    </>
  );
}

// ── Restarted PDHG ────────────────────────────────────────────────────────────────────────

const numOrNull = (v: unknown) => (typeof v === 'number' ? v : null);

/** The measures of a PDHG step, in a fixed grid so nothing moves while the numbers change. */
function PdhgPanel({ run, k }: { run: WorkRun; k: number }) {
  const i = run.trace[k].info as Record<string, unknown>;
  const cells: [string, string, string][] = [
    ['kkt', '\\text{KKT}(\\mathbf z_k)', texSci(numOrNull(i.kkt_last))],
    ['kkta', '\\text{KKT}(\\bar{\\mathbf z})', texSci(numOrNull(i.kkt_avg))],
    ['pri', '\\|A\\mathbf z_k - \\mathbf b\\|_{\\text{rel}}', texSci(numOrNull(i.primal_residual))],
    [
      'dual',
      '\\|(\\mathbf c - A^{\\top}\\mathbf y_k)_-\\|_{\\text{rel}}',
      texSci(numOrNull(i.dual_residual)),
    ],
    [
      'gap',
      '|\\mathbf c^{\\top}\\mathbf z_k - \\mathbf b^{\\top}\\mathbf y_k|_{\\text{rel}}',
      texSci(numOrNull(i.gap)),
    ],
    ['omega', '\\omega', texNum(numOrNull(i.omega), 3)],
    [
      'epoch',
      '\\text{epoch } n,\\ \\text{step } t',
      `${String(i.epoch)},\\ ${String(i.epoch_len)}`,
    ],
    [
      'rho',
      '\\rho(\\bar{\\mathbf z}) \\le \\beta\\rho_0',
      i.normalized_gap === null || i.normalized_gap === undefined
        ? '\\text{—}'
        : `${texSci(numOrNull(i.normalized_gap))} \\mathrel{${i.restarted ? '\\le' : '>'}} ${texSci(numOrNull(i.restart_threshold))}`,
    ],
  ];
  return (
    <dl className={styles.pdhgStats}>
      {cells.map(([key, tex, v]) => (
        <div key={key}>
          <dt>
            <F tex={tex} />
          </dt>
          <dd>
            <F tex={v} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Every sentence the PDHG strip can show, with this step's numbers. All are laid out in one grid
 * cell and only the current one is visible, so the strip keeps its height during playback.
 */
function PdhgNote({ run, k }: { run: WorkRun; k: number }) {
  const step = run.trace[k];
  const i = step.info as Record<string, unknown>;
  const n = Number(i.epoch);
  const t = Number(i.epoch_len);
  const tol = Number(run.params.tol ?? 1e-8);
  const scheme = String(run.params.restart ?? 'adaptive');
  const last = k === run.trace.length - 1;
  const kktLast = numOrNull(i.kkt_last) ?? Infinity;
  const kktAvg = numOrNull(i.kkt_avg) ?? Infinity;
  const which = kktAvg < kktLast ? 'average' : 'iterate';
  const restartWhy =
    scheme === 'fixed' ? (
      <>after the fixed period of {t} steps</>
    ) : n === 1 ? (
      <>at the first test (step {t} of the first epoch)</>
    ) : (
      <>
        : the normalized gap of the average fell to{' '}
        <F
          tex={`\\rho = ${texNum(numOrNull(i.normalized_gap), 3)} \\le \\beta\\rho_0 = ${texNum(numOrNull(i.restart_threshold), 3)}`}
        />
      </>
    );
  const variants: [string, ReactNode][] = [
    [
      'start',
      <>
        Start at <F tex={`\\mathbf x_0 = ${texVec(step.x as number[])}`} /> with{' '}
        <F tex="\mathbf y_0 = \mathbf 0" />. Each step costs one product with <F tex="A" /> and one
        with <F tex="A^{\top}" />; no linear system is solved. Squares mark the restarts.
      </>,
    ],
    [
      'step',
      <>
        Step {t} of epoch {n}: a projected step down in <F tex="\mathbf z" />, then up in{' '}
        <F tex="\mathbf y" />. The iterate circles the saddle point; the epoch average (ring) is
        steadier, with KKT error <F tex={texNum(kktAvg, 3)} /> against{' '}
        <F tex={texNum(kktLast, 3)} />.
      </>,
    ],
    [
      'check',
      <>
        Restart test at step {t} of epoch {n}: the normalized gap of the average,{' '}
        <F tex={`\\rho = ${texNum(numOrNull(i.normalized_gap), 3)}`} />, is still above{' '}
        <F tex={`\\beta\\rho_0 = ${texNum(numOrNull(i.restart_threshold), 3)}`} />, so the epoch
        goes on.
      </>,
    ],
    [
      'restart',
      <>
        Restart {n}
        {scheme === 'fixed' || n === 1 ? ' ' : ''}
        {restartWhy}. The next epoch starts from the average (square), not from the last iterate
        (cross).
      </>,
    ],
    [
      'final',
      run.converged ? (
        <>
          Converged in {k} steps and {n} restarts: the relative KKT error of the {which} is{' '}
          <F tex={`${texNum(Math.min(kktLast, kktAvg), 3)} \\le ${texNum(tol)}`} />.
        </>
      ) : (
        <>
          Stopped at the {k}-step budget: the KKT error is{' '}
          <F tex={`${texNum(Math.min(kktLast, kktAvg), 3)} > ${texNum(tol)}`} />. This PDHG has no
          infeasibility test, so an infeasible or unbounded LP runs to the budget.
        </>
      ),
    ],
  ];
  const active =
    k === 0
      ? 'start'
      : last
        ? 'final'
        : i.restarted === true
          ? 'restart'
          : i.normalized_gap !== null && i.normalized_gap !== undefined
            ? 'check'
            : 'step';
  return (
    <span className={styles.noteStack}>
      {variants.map(([key, node]) => (
        <span key={key} aria-hidden={key === active ? undefined : true}>
          {node}
        </span>
      ))}
    </span>
  );
}

// ── The strip ─────────────────────────────────────────────────────────────────────────────

export function WorkStrip({
  run,
  k,
  lp,
  compact = false,
}: {
  run: WorkRun;
  k: number;
  lp: LinearProgram;
  /** Narrow screens: fewer digits in the tableau. */
  compact?: boolean;
}) {
  const step = run.trace[k];
  const n = run.trace.length;
  const hasTableau = Array.isArray(step?.info.tableau);
  const isTree = run.id === 'branch_and_bound';
  const isPdhg = run.id === PDHG;
  const renderNote = useCallback(
    (i: number) =>
      hasTableau ? (
        <TableauNote run={run} k={i} />
      ) : isTree ? (
        <TreeNote run={run} k={i} lp={lp} />
      ) : (
        <InteriorNote run={run} k={i} />
      ),
    [run, lp, hasTableau, isTree],
  );
  // Reserve the width of the widest step number and of " · phase 1" (when the run has a phase 1),
  // so the title does not change width during playback.
  const kWidth = `calc(${String(Math.max(0, n - 1)).length} * 0.5 * 1.21em)`; // KaTeX digits: 0.5 em
  const hasPhase1 = useMemo(() => run.trace.some((s) => s.info.phase === 1), [run.trace]);
  if (!step) return null;
  const title = hasTableau
    ? 'Tableau'
    : isTree
      ? 'Search tree'
      : isPdhg
        ? 'Primal–dual measures'
        : 'Central-path measures';
  return (
    <section className={styles.strip} aria-label={`${title} of ${run.name}`}>
      <header className={styles.stripHead}>
        <h2 className={styles.stripTitle}>
          {title}
          <span className={styles.stripStep}>
            {' '}
            · <F tex="k =" />{' '}
            <span className={styles.stripK} style={{ minWidth: kWidth }}>
              <F tex={String(k)} />
            </span>
            {hasPhase1 && (
              <span
                aria-hidden={step.info.phase === 1 ? undefined : true}
                data-off={step.info.phase === 1 ? undefined : ''}
              >
                {' · phase 1'}
              </span>
            )}
          </span>
        </h2>
      </header>
      <p className={styles.stripNote}>
        {isPdhg ? <PdhgNote run={run} k={k} /> : <NoteStack n={n} k={k} render={renderNote} />}
      </p>
      <div className={styles.stripBody}>
        {hasTableau ? (
          <TableauPanel run={run} k={k} compact={compact} />
        ) : isTree ? (
          <TreePanel run={run} k={k} />
        ) : isPdhg ? (
          <PdhgPanel run={run} k={k} />
        ) : (
          <InteriorPanel run={run} k={k} />
        )}
      </div>
    </section>
  );
}
