/**
 * The lab's "Table" tab: the data structure each method builds, revealed up to the playhead.
 *
 *   Newton      the divided-difference table, one row per node (Burden & Faires, Alg. 3.2)
 *   Neville     the tableau Q_{i,j}(x⋆), one column per step (Alg. 3.1)
 *   splines     the tridiagonal system T s = r, then U s = z after the sweep, then s
 *   PCHIP       secants, slopes and the Fritsch–Carlson ratios α, β per interval
 *   Chebyshev   the coefficients cₖ and their decay
 *   Lagrange / barycentric   the nodes with max|ℓⱼ| or the weights wⱼ
 *   AAA         the support points zₘ in the order chosen, their weights at step k, and the
 *               poles of r_k with their residues (NST 2018, §5: Froissart doublets)
 *   Floater–Hormann   the weights wₖ and the windows Jₖ of local polynomials (FH 2007, eq. 18)
 */
import { useEffect, useMemo, useRef } from 'react';
import type { Result } from '../../core/types';
import { sig, sigFixed } from '../../core/format';
import { Formula } from '../../ui/components';
import { TableView, type TableColumn } from '../../viz';
import { aaaIntervalPoles, aaaPoles, basisFn, maxAbs, methodKind } from './geometry';
import { mathHead } from './heads';
import styles from './InterpolationLab.module.css';

const nums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);
const cell = (v: number | null | undefined) => sigFixed(v ?? null, 5);
/** Four significant digits: the narrow insight column holds six of these in a row. */
const cell4 = (v: number | null | undefined) => sigFixed(v ?? null, 4);

interface TableProps {
  methodId: string;
  result: Result;
  k: number;
  nodes: readonly number[];
  values: readonly number[];
  domain: [number, number];
  onSelect?: (k: number) => void;
}

/**
 * A lower-triangular table (row i has i + 1 entries). `mode` says what one step fills:
 * a row (Newton: node i brings row i) or a column (Neville: step k fills column k for every
 * row). The current row or column gets the accent wash; only what is not yet computed is faint.
 */
function TriangleTable({
  rows,
  current,
  mode,
  diagonal,
  head,
  headLabel,
  rowLabel,
  caption,
  onSelect,
}: {
  rows: (number | null)[][];
  current: number;
  mode: 'row' | 'column';
  /** Mark entry j of row j (the Newton coefficients / Neville estimates). */
  diagonal: boolean;
  head: (j: number) => string;
  /** Spoken header text (the KaTeX is not read as a header name). */
  headLabel: (j: number) => string;
  rowLabel: (i: number) => string;
  caption: string;
  onSelect?: (k: number) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const width = Math.max(...rows.map((r) => r.length), 1);
  const byRow = mode === 'row';
  useEffect(() => {
    const el = scroller.current?.querySelector<HTMLElement>('[data-current]');
    el?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [current]);
  return (
    <div className={styles.triWrap} ref={scroller} tabIndex={0} role="region" aria-label={caption}>
      <table className={styles.tri}>
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr>
            <th scope="col" className={styles.triIndex}>
              <Formula tex="i" />
              <span className="visually-hidden">row i</span>
            </th>
            {Array.from({ length: width }, (_, j) => (
              <th
                key={j}
                scope="col"
                data-current={(!byRow && j === current) || undefined}
                data-future={(!byRow && j > current) || undefined}
                onClick={!byRow && onSelect ? () => onSelect(j) : undefined}
              >
                <Formula tex={head(j)} />
                <span className="visually-hidden">{headLabel(j)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              data-current={(byRow && i === current) || undefined}
              data-future={(byRow && i > current) || undefined}
              onClick={byRow && onSelect ? () => onSelect(i) : undefined}
            >
              <th scope="row" className={styles.triIndex}>
                <Formula tex={rowLabel(i)} />
                <span className="visually-hidden">row {i}</span>
              </th>
              {Array.from({ length: width }, (_, j) => (
                <td
                  key={j}
                  data-diag={
                    (diagonal && j === i && row[j] !== undefined && (byRow || j <= current)) ||
                    undefined
                  }
                  data-col={(!byRow && j === current && row[j] !== undefined) || undefined}
                  data-future={(!byRow && j > current && row[j] !== undefined) || undefined}
                  onClick={
                    !byRow && onSelect && row[j] !== undefined ? () => onSelect(j) : undefined
                  }
                >
                  {row[j] === undefined ? '' : !byRow && j > current ? '·' : cell(row[j])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NewtonTable({ result, k, nodes, onSelect }: TableProps) {
  const rows = result.trace.map((s, i) => (i <= k ? nums(s.info.new_row) : []));
  const ord = (j: number) =>
    j === 0 ? 'f[x_i]' : j === 1 ? 'f[x_{i-1},x_i]' : `f[x_{i-${j}},\\dots,x_i]`;
  return (
    <>
      <p className={styles.tableNote}>
        Row <Formula tex="i" /> needs only node <Formula tex="x_i" /> and row <Formula tex="i-1" />.
        The underlined diagonal holds the coefficients <Formula tex="a_i = f[x_0,\dots,x_i]" />.
      </p>
      <div
        className={styles.tableFormula}
        tabIndex={0}
        role="region"
        aria-label="Divided-difference recursion"
      >
        <Formula
          display
          fit
          tex="f[x_{i-j},\dots,x_i] = \frac{f[x_{i-j+1},\dots,x_i] - f[x_{i-j},\dots,x_{i-1}]}{x_i - x_{i-j}}"
        />
      </div>
      <TriangleTable
        rows={rows.map((r, i) => (i <= k ? r : []))}
        current={k}
        mode="row"
        diagonal
        head={ord}
        headLabel={(j) => `divided difference of order ${j}`}
        rowLabel={(i) => `${i}\\;\\; x_{${i}} = ${sig(nodes[i], 4).replace('−', '-')}`}
        caption="Divided-difference table"
        onSelect={onSelect}
      />
    </>
  );
}

function NevilleTable({ result, k, nodes, onSelect }: TableProps) {
  const n = nodes.length;
  const cols = result.trace.map((s) => nums(s.info.column));
  // Q_{i,j} = column j, entry i − j (Python's info.column lists Q_{j,j}, Q_{j+1,j}, …).
  // Columns after the playhead are not computed yet and stay empty.
  const rows = Array.from({ length: n }, (_, i) =>
    Array.from({ length: i + 1 }, (_, j) => (j <= k ? (cols[j]?.[i - j] ?? null) : null)),
  );
  const xs = result.trace[0]?.info.x_eval as number | undefined;
  return (
    <>
      <p className={styles.tableNote}>
        <Formula tex="Q_{i,j} = P_{i-j,\dots,i}(x^\ast)" />, the interpolant through{' '}
        <Formula tex="j+1" /> consecutive nodes at{' '}
        <Formula tex={`x^\\ast = ${sig(xs ?? NaN, 4).replace('−', '-')}`} />. Step{' '}
        <Formula tex="k" /> fills column <Formula tex="k" />; the diagonal is{' '}
        <Formula tex="P_{0..k}(x^\ast)" />.
      </p>
      <TriangleTable
        rows={rows}
        current={k}
        mode="column"
        diagonal
        head={(j) => `Q_{i,${j}}`}
        headLabel={(j) => `Q i ${j}, degree ${j}`}
        rowLabel={(i) => `${i}`}
        caption="Neville tableau at x star"
        onSelect={onSelect}
      />
    </>
  );
}

function SplineTable({ result, k, onSelect }: TableProps) {
  const s0 = result.trace[0]?.info ?? {};
  const s1 = result.trace[1]?.info ?? {};
  const s2 = result.trace[2]?.info ?? {};
  const lower = nums(s0.lower),
    diag = nums(s0.diag),
    upper = nums(s0.upper),
    rhs = nums(s0.rhs);
  const n = diag.length;
  if (!n) return null;
  const stage = Math.min(k, 3);
  const piv = nums(s1.pivots),
    mult = nums(s1.multipliers),
    z = nums(s1.rhs),
    sl = nums(s2.slopes);
  type Row = { i: number };
  const swept = stage >= 1;
  const columns: TableColumn<Row>[] = [
    { key: 'i', ...mathHead('i'), value: (r) => r.i, width: '2rem' },
    {
      key: 'l',
      ...mathHead(swept ? 'l_i' : 'T_{i,i-1}'),
      value: (r) => (r.i === 0 ? '' : cell4(swept ? mult[r.i - 1] : lower[r.i - 1])),
    },
    {
      key: 'd',
      ...mathHead(swept ? 'u_i' : 'T_{ii}'),
      value: (r) => cell4(swept ? piv[r.i] : diag[r.i]),
    },
    { key: 'u', ...mathHead('T_{i,i+1}'), value: (r) => (r.i === n - 1 ? '' : cell4(upper[r.i])) },
    {
      key: 'r',
      ...mathHead(swept ? 'z_i' : 'r_i'),
      value: (r) => cell4(swept ? z[r.i] : rhs[r.i]),
    },
    { key: 's', ...mathHead('s_i'), value: (r) => (stage >= 2 ? cell4(sl[r.i]) : '·') },
  ];
  const caption =
    stage === 0
      ? `Assemble the tridiagonal system T s = r for the slopes sᵢ = S′(xᵢ); end rows: ${s0.boundary as string}.`
      : stage === 1
        ? 'Forward sweep (Thomas): lᵢ = Tᵢ,ᵢ₋₁/uᵢ₋₁ eliminates the subdiagonal; uᵢ are the pivots and z the updated right-hand side.'
        : stage === 2
          ? 'Back substitution: sₙ₋₁ = zₙ₋₁/uₙ₋₁, then sᵢ = (zᵢ − Tᵢ,ᵢ₊₁ sᵢ₊₁)/uᵢ, last row first.'
          : 'The slopes give each piece in Hermite form; see the curve.';
  return (
    <>
      <p className={styles.tableNote}>{caption}</p>
      <TableView
        maxHeight="none"
        columns={columns}
        rows={Array.from({ length: n }, (_, i) => ({ i }))}
        onSelect={onSelect ? () => onSelect(Math.min(stage + 1, 3)) : undefined}
        ariaLabel="Tridiagonal spline system, one row per equation"
      />
    </>
  );
}

function PchipTable({ result, k, nodes, onSelect }: TableProps) {
  const s0 = result.trace[0]?.info ?? {};
  const s1 = result.trace[1]?.info ?? {};
  const m = nums(s0.secants);
  const slopes = k >= 1 ? nums(s1.slopes) : [];
  const limited = k >= 1 ? ((s1.limited as boolean[] | undefined) ?? []) : [];
  const alpha = k >= 1 ? ((s1.alpha as (number | null)[] | undefined) ?? []) : [];
  const beta = k >= 1 ? ((s1.beta as (number | null)[] | undefined) ?? []) : [];
  type Row = { i: number };
  const rows: Row[] = nodes.map((_, i) => ({ i }));
  const ab = (v: number | null | undefined) => (v === null ? '—' : v === undefined ? '' : cell4(v));
  const columns: TableColumn<Row>[] = [
    { key: 'i', ...mathHead('i'), value: (r) => r.i, width: '2rem' },
    { key: 'm', ...mathHead('m_i'), value: (r) => (r.i < m.length ? cell4(m[r.i]) : '') },
    {
      key: 's',
      ...mathHead('s_i'),
      value: (r) => (slopes.length ? `${cell4(slopes[r.i])}${limited[r.i] ? ' ◦' : ''}` : '·'),
    },
    {
      key: 'ab',
      ...mathHead('(\\alpha_i,\\ \\beta_i)', 'alpha i, beta i'),
      value: (r) => (r.i < alpha.length ? `${ab(alpha[r.i])}, ${ab(beta[r.i])}` : ''),
    },
  ];
  return (
    <>
      <p className={styles.tableNote}>
        Secants <Formula tex="m_i" />, node slopes <Formula tex="s_i" /> (◦ = set by a shape rule),
        and <Formula tex="\alpha_i = s_i/m_i,\ \beta_i = s_{i+1}/m_i" />: the piece on{' '}
        <Formula tex="[x_i, x_{i+1}]" /> is monotone when both lie in <Formula tex="[0, 3]" />{' '}
        (Fritsch–Carlson).
      </p>
      <TableView
        maxHeight="none"
        columns={columns}
        rows={rows}
        onSelect={onSelect ? () => onSelect(Math.min(k + 1, 2)) : undefined}
        ariaLabel="PCHIP slopes"
      />
    </>
  );
}

function LinearSplineTable({ result, nodes }: TableProps) {
  const info = result.trace[0]?.info ?? {};
  const h = nums(info.h),
    m = nums(info.secants);
  const rows = h.map((_, i) => ({ i }));
  const columns: TableColumn<{ i: number }>[] = [
    { key: 'i', ...mathHead('i'), value: (r) => r.i, width: '2rem' },
    {
      key: 'x',
      ...mathHead('[x_i, x_{i+1}]'),
      value: (r) => `${sig(nodes[r.i], 4)} … ${sig(nodes[r.i + 1], 4)}`,
    },
    { key: 'h', ...mathHead('h_i'), value: (r) => cell(h[r.i]) },
    { key: 'm', ...mathHead('m_i'), value: (r) => cell(m[r.i]) },
  ];
  return (
    <TableView maxHeight="none" columns={columns} rows={rows} ariaLabel="Linear spline pieces" />
  );
}

function ChebTable({ result, k, onSelect }: TableProps) {
  const c = nums(result.trace[result.trace.length - 1]?.x);
  const rows = c.map((v, i) => ({ i, v }));
  const logs = c.map((v) => Math.log10(Math.max(Math.abs(v), 1e-17)));
  const top = Math.max(...logs, -16);
  const columns: TableColumn<{ i: number; v: number }>[] = [
    { key: 'k', ...mathHead('k'), value: (r) => r.i, width: '2rem' },
    { key: 'c', ...mathHead('c_k'), value: (r) => cell(r.v) },
    {
      key: 'bar',
      ...mathHead('\\log_{10}|c_k|', 'log 10 of |c k|'),
      align: 'left',
      mono: false,
      value: (r) => (
        <span className={styles.bar} aria-label={`|c| = ${sig(Math.abs(r.v), 2)}`}>
          <span style={{ width: `${Math.max(2, ((logs[r.i] + 17) / (top + 17)) * 100)}%` }} />
        </span>
      ),
    },
  ];
  const source = result.extra.source;
  return (
    <>
      <p className={styles.tableNote}>
        {source === 'f_true'
          ? 'f sampled at the roots of T_N; cₖ by discrete orthogonality. For analytic f, |cₖ| decays geometrically: the bars shrink at a constant rate.'
          : 'The data are not samples of f, so cₖ solves the Chebyshev–Vandermonde system through the data points.'}
      </p>
      <TableView
        maxHeight="none"
        columns={columns}
        rows={rows}
        highlight={k}
        futureAfter={k}
        onSelect={onSelect}
        ariaLabel="Chebyshev coefficients"
      />
    </>
  );
}

function NodeTable({ methodId, result, k, nodes, values, domain, onSelect }: TableProps) {
  const isBary = methodId === 'barycentric';
  const w = isBary ? nums(result.trace[Math.min(k, result.trace.length - 1)]?.x) : [];
  const rows = nodes.map((_, i) => ({ i }));
  // n basis maxima on 801 points: O(n²) work per node set, not per playback frame.
  const nodeKey = nodes.join(',');
  const lmax = useMemo(
    () =>
      isBary ? [] : nodes.map((_, j) => maxAbs(basisFn(nodes, j), domain[0], domain[1], 801).value),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isBary, nodeKey, domain[0], domain[1]],
  );
  const columns: TableColumn<{ i: number }>[] = [
    { key: 'i', ...mathHead('j'), value: (r) => r.i, width: '2rem' },
    { key: 'x', ...mathHead('x_j'), value: (r) => cell(nodes[r.i]) },
    { key: 'y', ...mathHead('y_j'), value: (r) => cell(values[r.i]) },
    isBary
      ? { key: 'w', ...mathHead('w_j'), value: (r) => (r.i < w.length ? cell(w[r.i]) : '') }
      : { key: 'l', ...mathHead('\\max|\\ell_j|'), value: (r) => cell(lmax[r.i]) },
  ];
  return (
    <>
      <p className={styles.tableNote}>
        {isBary
          ? 'When node k joins, every earlier weight is divided by (xⱼ − xₖ) and wₖ = 1/∏(xₖ − xₘ). The signs alternate on sorted nodes.'
          : 'ℓⱼ is 1 at xⱼ and 0 at the other nodes. Large max|ℓⱼ| (equispaced nodes near the ends) means a small change in yⱼ moves p far.'}
      </p>
      <TableView
        maxHeight="none"
        columns={columns}
        rows={rows}
        highlight={k}
        futureAfter={k}
        onSelect={onSelect}
        ariaLabel={isBary ? 'Barycentric weights' : 'Lagrange nodes'}
      />
    </>
  );
}

function AaaTable({ result, k, domain, onSelect }: TableProps) {
  const last = result.trace.length - 1;
  const kk = Math.min(k, last);
  const final = result.trace[last]?.info ?? {};
  const support = nums(final.support),
    fvals = nums(final.support_values);
  const w = nums(result.trace[kk]?.info.weights);
  const fScale = Math.max(...fvals.map(Math.abs), ...nums(result.extra.values).map(Math.abs), 0);
  type Row = { i: number };
  const columns: TableColumn<Row>[] = [
    { key: 'm', ...mathHead('m'), value: (r) => r.i + 1, width: '2rem' },
    { key: 'z', ...mathHead('z_m'), value: (r) => cell(support[r.i]) },
    { key: 'f', ...mathHead('f(z_m)', 'f of z m'), value: (r) => cell(fvals[r.i]) },
    { key: 'w', ...mathHead('w_m'), value: (r) => (r.i < w.length ? cell(w[r.i]) : '·') },
  ];
  // The poles of r_k: real ones in [a, b] are certified by a sign change of the denominator.
  const poles = aaaPoles(result, kk, fScale)
    .map((p) => ({ ...p, res: residueAbs(result, kk, p) }))
    .sort((p, q) => Math.abs(p.im) - Math.abs(q.im) || p.re - q.re);
  const certified = aaaIntervalPoles(result, kk);
  const tolX = 1e-6 * (domain[1] - domain[0]);
  type PoleRow = (typeof poles)[number];
  const poleCols: TableColumn<PoleRow>[] = [
    {
      key: 're',
      ...mathHead('\\operatorname{Re}\\lambda', 'real part of lambda'),
      value: (p) => cell4(p.re),
    },
    {
      key: 'im',
      ...mathHead('\\operatorname{Im}\\lambda', 'imaginary part of lambda'),
      value: (p) => (Math.abs(p.im) < 1e-12 * Math.max(1, Math.abs(p.re)) ? '0' : cell4(p.im)),
    },
    {
      key: 'res',
      ...mathHead('|\\operatorname{res}|', 'absolute residue'),
      value: (p) => (p.res === null ? '—' : sig(p.res, 2)),
    },
    {
      key: 'kind',
      header: 'Kind',
      align: 'left',
      mono: false,
      value: (p) =>
        p.doublet
          ? 'Froissart doublet'
          : certified.some((c) => Math.abs(c - p.re) <= tolX) &&
              Math.abs(p.im) <= 1e-6 * Math.max(1, Math.abs(p.re))
            ? 'real, in [a, b]'
            : Math.abs(p.im) <= 1e-9 * Math.max(1, Math.abs(p.re))
              ? 'real, outside [a, b]'
              : 'complex',
    },
  ];
  return (
    <>
      <p className={styles.tableNote}>
        Step <Formula tex="m" /> adds the sample with the largest error{' '}
        <Formula tex="|f - r_{m-1}|" /> as the support point <Formula tex="z_m" />. The weights are
        the smallest singular vector of the Loewner matrix and change at every step; row{' '}
        <Formula tex="m" /> shows <Formula tex="w_m" /> after step <Formula tex={String(kk)} />.
      </p>
      <TableView
        maxHeight="none"
        columns={columns}
        rows={support.map((_, i) => ({ i }))}
        highlight={kk - 1}
        futureAfter={kk - 1}
        onSelect={onSelect ? (i) => onSelect(i + 1) : undefined}
        ariaLabel="AAA support points and weights"
        empty="Step 0: no support point yet; r is the mean of the samples."
      />
      <p className={styles.tableNote} style={{ marginTop: 'var(--space-3)' }}>
        Poles of <Formula tex={`r_{${kk}}`} />:{' '}
        {poles.length === 0 ? 'none.' : `${poles.length}, nearest the real axis first. `}
        {certified.length > 0 &&
          `${certified.length} certified real pole${certified.length === 1 ? '' : 's'} in [a, b]. `}
        A pole with a residue below <Formula tex="10^{-13}\max|f|" /> is a Froissart doublet: a pole
        and a zero that almost cancel.
      </p>
      {poles.length > 0 && (
        <TableView
          maxHeight="none"
          columns={poleCols}
          rows={poles}
          ariaLabel={`Poles of r ${kk}`}
        />
      )}
    </>
  );
}

/** |residue| of the pole p at step k (matched by position in info.poles). */
function residueAbs(result: Result, k: number, p: { re: number; im: number }): number | null {
  const info = result.trace[k]?.info ?? {};
  const poles = (info.poles as number[][] | undefined) ?? [];
  const res = (info.residues as number[][] | undefined) ?? [];
  const i = poles.findIndex((q) => q[0] === p.re && q[1] === p.im);
  return i >= 0 && res[i] ? Math.hypot(res[i][0], res[i][1]) : null;
}

function BlendTable({ result, k, nodes, values, onSelect }: TableProps) {
  const d = result.extra.d as number | undefined;
  const w = nums(result.trace[result.trace.length - 1]?.x);
  const rows = nodes.map((_, i) => ({ i }));
  const columns: TableColumn<{ i: number }>[] = [
    { key: 'k', ...mathHead('k'), value: (r) => r.i, width: '2rem' },
    { key: 'x', ...mathHead('x_k'), value: (r) => cell(nodes[r.i]) },
    { key: 'y', ...mathHead('y_k'), value: (r) => cell(values[r.i]) },
    { key: 'w', ...mathHead('w_k'), value: (r) => (r.i <= k ? cell(w[r.i]) : '·') },
    {
      key: 'j',
      ...mathHead('J_k'),
      value: (r) => {
        const win = nums(result.trace[r.i]?.info.window);
        return win.length === 2 ? `${win[0]} … ${win[1]}` : '';
      },
    },
  ];
  return (
    <>
      <p className={styles.tableNote}>
        <Formula tex="w_k" /> sums over the window <Formula tex="J_k" />: the local polynomials{' '}
        <Formula tex="p_i" /> of degree <Formula tex={`d = ${d ?? 'd'}`} /> through{' '}
        <Formula tex="x_i, \dots, x_{i+d}" /> that contain node <Formula tex="k" />. The signs
        alternate, so <Formula tex="r" /> has no real pole. Every weight carries the common factor{' '}
        <Formula tex="h^d" /> (
        <Formula tex="h = (x_n - x_0)/n" />
        ), which cancels in <Formula tex="r" />.
      </p>
      <TableView
        maxHeight="none"
        columns={columns}
        rows={rows}
        highlight={k}
        futureAfter={k}
        onSelect={onSelect}
        ariaLabel="Floater–Hormann weights"
      />
    </>
  );
}

/** The method's own table at step k. */
export function MethodTable(props: TableProps) {
  const { methodId, result } = props;
  if (!result.trace.length) return <p className={styles.tableNote}>No steps to show.</p>;
  if (methodId === 'newton_divided_differences') return <NewtonTable {...props} />;
  if (methodId === 'neville') return <NevilleTable {...props} />;
  if (methodId === 'chebyshev_interpolation') return <ChebTable {...props} />;
  if (methodId === 'pchip') return <PchipTable {...props} />;
  if (methodId === 'linear_spline') return <LinearSplineTable {...props} />;
  if (methodId === 'aaa') return <AaaTable {...props} />;
  if (methodId === 'floater_hormann') return <BlendTable {...props} />;
  if (methodKind(methodId) === 'piecewise') return <SplineTable {...props} />;
  return <NodeTable {...props} />;
}
