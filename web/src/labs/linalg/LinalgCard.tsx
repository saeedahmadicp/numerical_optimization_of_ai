/**
 * The lab's textbook page for the current step (portal-direction §3), specialised to A x = b:
 * the update rule, the same rule with this step's numbers filled in, the quantities that
 * explain the step (the pivot and multipliers of a stage; for an iteration the observed
 * contraction ‖r_k‖/‖r_{k−1}‖ beside ρ(G), or CG's α, β and the conjugacy p_kᵀA p_{k−1}), the
 * outcome with its evidence, the intuition, when it fails, and the sources.
 *
 * The shared MethodCard labels Step.fun as f(x_k); here it is the residual ‖b − Ax_k‖₂, so the
 * lab keeps its own card.
 */
import { useMemo, type ReactNode } from 'react';
import type { RegisteredMethod } from '../../core/registry';
import { defaults } from '../../core/registry';
import type { Matrix, Params, Result, Step, Vector } from '../../core/types';
import { int, sci, sig } from '../../core/format';
import { CopyButton, Formula, Menu, MethodChip, type MenuItem } from '../../ui/components';
import { StepBlock, pythonCall } from '../_shell';
import { describe, msgText } from './messages';
import { MathProse } from './MathProse';
import { isDirect, isStationary, stageOf } from './model';
import { pivotSymbol, texNum } from './texfmt';
import styles from './LinalgCard.module.css';

export interface LinalgCardProps {
  method: RegisteredMethod;
  slot: number;
  result: Result;
  error?: string;
  /** The run's trace and the step under the playhead. */
  trace: readonly Step[];
  k: number;
  A: Matrix;
  n: number;
  problemId: string;
  params: Params;
  x0?: number[];
}

const show = (v: unknown): string => {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number')
    return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? sci(v, 3) : sig(v, 5);
  return String(v);
};

const vecTex = (v: readonly number[], digits = 4) =>
  `(${v.map((x) => texNum(x, digits)).join(',\\ ')})`;

function dotA(A: Matrix, u: Vector, w: Vector): number {
  let s = 0;
  for (let i = 0; i < u.length; i++) for (let j = 0; j < w.length; j++) s += u[i] * A[i][j] * w[j];
  return s;
}

/** This step's rule with the numbers in, and the live quantities. */
function thisStep(
  id: string,
  step: Step,
  prev: Step | undefined,
  back: Step | undefined,
  m: number,
  A: Matrix,
  n: number,
  params: Params,
  extra: Record<string, unknown>,
): { tex: string | null; cells: { tex: string; value: string }[] } {
  const info = step.info;
  const cells: { tex: string; value: string }[] = [{ tex: 'k', value: int(step.k) }];
  if (isDirect(id)) {
    const s = stageOf(step, n);
    cells.push({ tex: '\\text{phase}', value: s.phase.replace('_', ' ') });
    if (s.phase === 'eliminate' && s.pivot) {
      const c = s.pivot[1] + 1;
      cells.push({ tex: '\\text{pivot}', value: show(s.pivotValue) });
      if (s.rowSwap)
        cells.push({ tex: '\\text{swap}', value: `R${s.rowSwap[0] + 1} ↔ R${s.rowSwap[1] + 1}` });
      const p = texNum(s.pivotValue);
      if (s.zeroPivot)
        return {
          tex: `|${pivotSymbol(id, c - 1)}| = ${texNum(Math.abs(s.pivotValue ?? 0), 3)} \\le \\tau:\\ \\text{stop}`,
          cells,
        };
      const m = s.multipliers?.slice(c) ?? [];
      switch (id) {
        case 'cholesky':
          return {
            tex: `l_{${c}${c}} = \\sqrt{w_{${c}${c}}} = \\sqrt{${p}} = ${texNum(Math.sqrt(s.pivotValue ?? NaN))}`,
            cells,
          };
        case 'qr_householder':
          return {
            tex: `r_{${c}${c}} = -\\operatorname{sign}(x_1)\\|\\mathbf{x}\\|_2 = ${p}`,
            cells,
          };
        case 'thomas':
          return {
            tex: `w_{${c}} = ${p},\\quad d'_{${c}} = ${texNum((info.d_prime as number[] | undefined)?.[c - 1])}`,
            cells,
          };
        case 'lu_decomposition':
          return {
            tex: multTex('\\ell', 'w', c, n, p, m),
            cells,
          };
        case 'gauss_jordan':
          return {
            tex: `R_{${c}} \\leftarrow R_{${c}} / ${p.startsWith('-') ? `(${p})` : p}`,
            cells,
          };
        default:
          return { tex: multTex('m', 'a', c, n, p, m), cells };
      }
    }
    if (Array.isArray(step.x))
      return {
        tex: `\\|\\mathbf{b} - A\\mathbf{x}\\|_2 = ${texNum(step.fun, 3)}`,
        cells: [...cells, { tex: '\\eta_\\infty', value: show(extra.backward_error) }],
      };
    return { tex: null, cells };
  }

  cells.push(
    { tex: '\\|\\mathbf{r}_k\\|_2', value: show(step.fun) },
    { tex: '\\|\\mathbf{r}_k\\|/d', value: show(info.relative_residual) },
  );
  if (isStationary(id)) {
    const rho = extra.spectral_radius as number | undefined;
    // Observed contraction per step over the last m steps; it tends to ρ(G).
    const q = back?.fun && step.fun !== null && m > 0 ? (step.fun / back.fun) ** (1 / m) : null;
    cells.push({ tex: '\\rho(G)', value: show(rho) });
    if (id === 'sor') cells.push({ tex: '\\omega', value: show(params.omega) });
    const ratio =
      m === 1
        ? `\\frac{\\|\\mathbf{r}_{${step.k}}\\|}{\\|\\mathbf{r}_{${step.k - 1}}\\|}`
        : `\\Big(\\frac{\\|\\mathbf{r}_{${step.k}}\\|}{\\|\\mathbf{r}_{${step.k - m}}\\|}\\Big)^{1/${m}}`;
    return {
      tex:
        q !== null && Number.isFinite(q)
          ? `${ratio} = ${texNum(q, 4)}\\ \\xrightarrow{k\\to\\infty}\\ \\rho(G) = ${texNum(rho ?? NaN, 4)}`
          : `\\rho(G) = ${texNum(rho ?? NaN, 4)}`,
      cells,
    };
  }
  if (id === 'gmres') {
    cells.push(
      { tex: 'j', value: show(info.krylov_dim) },
      { tex: '\\text{cycle}', value: show(info.cycle) },
    );
    const g = info.givens as [number, number] | null;
    return {
      tex: g
        ? `|g_{j+1}| = ${texNum(step.fun, 3)},\\quad (c_j, s_j) = (${texNum(g[0], 3)},\\ ${texNum(g[1], 3)})`
        : `\\mathbf{r}_0 = \\mathbf{b} - A\\mathbf{x}_0,\\ \\beta = ${texNum(step.fun, 4)}`,
      cells,
    };
  }
  // SD, CG, PCG
  const alpha = info.alpha as number | null;
  cells.push(
    { tex: '\\alpha_{k-1}', value: show(alpha) },
    { tex: '\\varphi(\\mathbf{x}_k)', value: show(info.phi) },
  );
  if (id === 'steepest_descent_linear')
    return {
      tex:
        alpha !== null
          ? `\\alpha_{${step.k - 1}} = \\frac{\\mathbf{r}^{\\mathsf{T}}\\mathbf{r}}{\\mathbf{r}^{\\mathsf{T}}A\\mathbf{r}} = ${texNum(alpha)},\\quad \\mathbf{r}_{${step.k}}^{\\mathsf{T}}\\mathbf{r}_{${step.k - 1}} = ${texNum(prev ? dotV(info.residual as Vector, prev.info.residual as Vector) : NaN, 2)}`
          : `\\mathbf{r}_0 = \\mathbf{b} - A\\mathbf{x}_0`,
      cells,
    };
  const beta = info.beta as number | null;
  cells.push({ tex: '\\beta_k', value: show(beta) });
  const p = info.direction as Vector | undefined;
  const pPrev = prev?.info.direction as Vector | undefined;
  const conj = p && pPrev ? dotA(A, p, pPrev) : null;
  return {
    tex:
      alpha !== null
        ? `\\alpha_{${step.k - 1}} = ${texNum(alpha)},\\ \\beta_{${step.k}} = ${texNum(beta ?? NaN)},\\quad \\mathbf{p}_{${step.k}}^{\\mathsf{T}}A\\,\\mathbf{p}_{${step.k - 1}} = ${texNum(conj ?? NaN, 2)}`
        : `\\mathbf{p}_0 = ${id === 'preconditioned_cg' ? 'D^{-1}' : ''}\\mathbf{r}_0`,
    cells,
  };
}

/**
 * The multipliers of stage c (1-based): `m_{ic} = a_{ic}/a_{cc}` with the vector of values, or the
 * single multiplier as `m_{32} = −0.5` (LU writes ℓ and the working matrix w).
 */
function multTex(
  m: string,
  a: string,
  c: number,
  n: number,
  p: string,
  vals: readonly number[],
): string {
  const den = p.startsWith('-') ? `(${p})` : p;
  if (vals.length === 0) return `${a}_{${c}${c}} = ${p}`;
  const rule = `${m}_{i${c}} = \\frac{${a}_{i${c}}}{${a}_{${c}${c}}} = \\frac{${a}_{i${c}}}{${den}}\\quad (i > ${c})`;
  const values =
    vals.length === 1
      ? `${m}_{${c + 1}${c}} = ${texNum(vals[0], 3)}`
      : `(${m}_{${c + 1}${c}},\\dots,${m}_{${n}${c}}) = ${vecTex(vals, 3)}`;
  return `\\begin{gathered}${rule}\\\\ ${values}\\end{gathered}`;
}

const dotV = (a: Vector, b: Vector) => a.reduce((s, v, i) => s + v * b[i], 0);

function statusLine(id: string, result: Result, error: string | undefined, params: Params) {
  if (error) return { tone: 'bad' as const, icon: '×', text: `Could not run: ${msgText(error)}` };
  if (isDirect(id)) {
    if (result.converged) {
      const eta = result.extra.backward_error as number;
      const bound = result.extra.backward_error_bound as number;
      return {
        tone: 'good' as const,
        icon: '✓',
        text: `Solved in ${int(result.nIter)} steps · backward error η∞ = ${sci(eta, 2)} ≤ ${sci(bound, 2)}`,
      };
    }
    return { tone: 'bad' as const, icon: '△', text: msgText(result.message, params) };
  }
  const st = describe(result);
  return { tone: st.tone, icon: st.icon, text: st.long, why: msgText(result.message, params) };
}

/** The step-k inputs of `thisStep` (an even window back for the observed contraction). */
function stepAt(trace: readonly Step[], k: number) {
  const step: Step | undefined = trace[k];
  const prev = k > 0 ? trace[k - 1] : undefined;
  // An even window: Jacobi's ± eigenvalue pairs make one-step ratios alternate about ρ.
  const m = Math.min(4, k - (k % 2 === 1 && k > 1 ? 1 : 0));
  const back = m > 0 ? trace[k - m] : undefined;
  return { step, prev, back, m };
}

/** Numbers do not change a formula's height; its shape (fractions, rows, operators) does. */
const texShape = (tex: string) => tex.replace(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi, '0');

/**
 * The run's layout envelope: the longest step formula of each shape and the largest number of
 * quantity cells. Laid out invisibly with the current step, they keep the block and the grid at
 * one size for the whole run, so nothing below moves during playback.
 */
function runEnvelope(
  id: string,
  trace: readonly Step[],
  A: Matrix,
  n: number,
  params: Params,
  extra: Record<string, unknown>,
): { sizers: string[]; cells: number } {
  const byShape = new Map<string, string>();
  let cells = 0;
  for (let i = 0; i < trace.length; i++) {
    const { step, prev, back, m } = stepAt(trace, i);
    if (!step) continue;
    const now = thisStep(id, step, prev, back, m, A, n, params, extra);
    cells = Math.max(cells, now.cells.length);
    if (!now.tex) continue;
    const key = texShape(now.tex);
    const cur = byShape.get(key);
    if (!cur || now.tex.length > cur.length) byShape.set(key, now.tex);
  }
  return { sizers: [...byShape.values()], cells };
}

/** "Stage 2" (a stage of a direct method), "Start" (k = 0); otherwise StepBlock's "Step 3 → 4". */
function stepLabel(direct: boolean, k: number): string | undefined {
  if (direct) return `Stage ${k}`;
  return k === 0 ? 'Start' : undefined;
}

export function LinalgCard({
  method,
  slot,
  result,
  error,
  trace,
  k,
  A,
  n,
  problemId,
  params,
  x0,
}: LinalgCardProps) {
  const { step, prev, back, m } = stepAt(trace, k);
  const { spec, doc } = method;
  const order = doc?.order ?? spec.order;
  const all = { ...defaults(spec), ...params };
  const envelope = useMemo(
    () => runEnvelope(spec.id, trace, A, n, { ...defaults(spec), ...params }, result.extra),
    [spec, trace, A, n, params, result.extra],
  );
  const status = statusLine(spec.id, result, error, all);
  const now = step ? thisStep(spec.id, step, prev, back, m, A, n, all, result.extra) : null;
  const iterative = !isDirect(spec.id);
  const menu: MenuItem[] = [
    {
      kind: 'copy',
      label: 'Copy Python call',
      icon: 'code',
      text: () =>
        pythonCall(spec.id, {
          problem: problemId,
          x0: iterative && x0 ? x0 : undefined,
          params,
          specs: spec.params,
        }),
    },
    ...(doc?.rule
      ? [{ kind: 'copy' as const, label: 'Copy update rule (LaTeX)', text: doc.rule }]
      : []),
    ...(Array.isArray(step?.x)
      ? [
          {
            kind: 'copy' as const,
            label: 'Copy the current iterate',
            text: () => `[${(step!.x as number[]).join(', ')}]`,
          },
        ]
      : []),
    {
      kind: 'link',
      label: 'Find in the method catalog',
      icon: 'book',
      href: `#/methods?q=${encodeURIComponent(spec.id)}`,
    },
  ];
  return (
    <div className={styles.card} tabIndex={0} role="region" aria-label={`${spec.name} details`}>
      {/* The same head as the shared MethodCard: chip, then the rate and the menu at the end. */}
      <div className={styles.head}>
        <MethodChip name={spec.name} slot={slot} />
        <span className={styles.headEnd}>
          {order && (
            <span className={styles.rate}>
              <MathProse text={order} />
            </span>
          )}
          <Menu label={`More actions for ${spec.name}`} items={menu} />
        </span>
      </div>
      {doc?.rule && (
        <div className={styles.rule}>
          <Formula tex={doc.rule} display fit />
        </div>
      )}
      {envelope.sizers.length > 0 && (
        // The formula of step k is the move into 𝐱ₖ ("Step 3 → 4" at k = 4); the sizers (the
        // run's largest formulas) keep the block at one height from the first frame.
        <StepBlock
          k={Math.max(0, k - 1)}
          label={stepLabel(!iterative, k)}
          reset={`${spec.id}|${slot}`}
        >
          <div className={styles.filledStack}>
            <div>{now?.tex && <Formula tex={now.tex} display fit />}</div>
            {envelope.sizers.map((t) => (
              <div key={t} className={styles.sizer} aria-hidden="true">
                <Formula tex={t} display fit />
              </div>
            ))}
          </div>
        </StepBlock>
      )}
      {now && (
        <dl className={styles.live} aria-label="This step">
          {now.cells.map((q) => (
            <div key={q.tex} className={styles.liveCell}>
              <dt className={styles.liveLabel}>
                <Formula tex={q.tex} />
              </dt>
              <dd className={styles.liveValue}>{q.value}</dd>
            </div>
          ))}
          {/* Empty cells up to the run's largest count: the grid keeps its rows. */}
          {Array.from({ length: Math.max(0, envelope.cells - now.cells.length) }, (_, i) => (
            <div key={`pad${i}`} className={styles.liveCell} aria-hidden="true" />
          ))}
        </dl>
      )}
      <div className={styles.statusBlock}>
        <p className={styles.status} data-tone={status.tone}>
          <span className={styles.statusIcon} aria-hidden="true">
            {status.icon}
          </span>
          {status.text}
        </p>
        {'why' in status && status.why && status.tone !== 'bad' && (
          <p className={styles.evidence}>{status.why}</p>
        )}
      </div>
      {(doc?.intuition || spec.summary) && (
        <p className={styles.intuition}>
          <MathProse text={doc?.intuition ?? spec.summary} />
        </p>
      )}
      {(doc?.pros?.length || doc?.cons?.length) && (
        <div className={styles.proscons}>
          <List title="Strengths" items={doc?.pros} className={styles.pro} />
          <List title="When it fails" items={doc?.cons} className={styles.con} />
        </div>
      )}
      {spec.references.length > 0 && (
        <div className={styles.refs}>
          <h3 className="visually-hidden">Sources</h3>
          {spec.references.map((r) => (
            <p key={r} className={styles.ref}>
              <cite>{r}</cite>
              <CopyButton text={r} label={`Copy the reference: ${r}`} className={styles.refCopy}>
                <span className="visually-hidden">Copy</span>
              </CopyButton>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function List({
  title,
  items,
  className,
}: {
  title: string;
  items?: string[];
  className: string;
}): ReactNode {
  if (!items?.length) return <div />;
  return (
    <div className={className}>
      <h3>{title}</h3>
      <ul>
        {items.map((p) => (
          <li key={p}>
            <MathProse text={p} />
          </li>
        ))}
      </ul>
    </div>
  );
}
