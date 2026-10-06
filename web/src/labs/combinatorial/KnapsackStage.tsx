/**
 * Knapsack stage: the items profile (bars of width wᵢ and height vᵢ/wᵢ in ratio order; the area
 * left of the capacity line is the Dantzig LP bound) with one knapsack per method below it, and
 * the detail of the method in focus — the DP table filling cell by cell with its backtrack path,
 * or the branch-and-bound tree. The capacity line can be dragged (or moved with the arrow keys
 * when the profile has focus) to change C; every method re-runs.
 */
import {
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from 'react';
import { Formula } from '../../ui/components';
import type { TracePlayer } from '../../play/useTracePlayer';
import { useCanvas } from '../../viz';
import { useChartColors } from '../../ui/theme';
import { sig } from '../../core/format';
import type { Step } from '../../core/types';
import type { KnapsackInstance } from '../../problems/combinatorial';
import type { BbNode } from '../../methods/combinatorial/knapsack';
import type { LabRun } from '../_shell';
import { bbFixings, bbNodesUpTo, dpBacktrackPath, selectedItems } from './model';
import {
  dpCellAt,
  drawDpTable,
  drawItems,
  itemAt,
  type DpLayout,
  type FocusMarks,
  type ItemsLayout,
} from './knapsackDraw';
import { BbTree } from './BbTree';
import styles from './CombinatorialLab.module.css';

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (n: number) => [...String(n)].map((c) => SUBS[Number(c)]).join('');

const SHORT: Record<string, string> = {
  knapsack_dp: 'DP',
  knapsack_greedy: 'Greedy',
  knapsack_branch_bound: 'B&B',
};

export type KnapsackView = 'both' | 'items' | 'detail';

export interface KnapsackStageProps {
  problem: KnapsackInstance;
  runs: readonly LabRun[];
  player: TracePlayer;
  focus: LabRun | undefined;
  focusIndex: number;
  reference: number | null;
  onCapacity: (C: number) => void;
  /** What to show (phones show one at a time). */
  view: KnapsackView;
}

/** Marks of the focus method's current state on the items profile. */
function focusMarks(run: LabRun, step: Step | undefined, all: BbNode[]): FocusMarks | null {
  if (!step) return null;
  const id = run.sel.id;
  const packed = new Set(selectedItems(step.x));
  const excluded = new Set<number>();
  let cursor: number | null = null;
  let cursorLabel: string | undefined;
  if (id === 'knapsack_greedy') {
    const order = (step.info.order as number[] | undefined) ?? [];
    const examined = step.info.phase === 'start' ? 0 : step.k;
    for (let s = 0; s < Math.min(examined, order.length); s++)
      if (!packed.has(order[s])) excluded.add(order[s]);
    cursor = (step.info.item as number | null | undefined) ?? null;
    if (cursor !== null)
      cursorLabel =
        step.info.phase === 'single_item_fix'
          ? `best single item ${cursor}`
          : `item ${cursor}: ${step.info.fits ? 'fits' : 'does not fit'}`;
  } else if (id === 'knapsack_branch_bound') {
    const nodes = (step.info.nodes as BbNode[]) ?? [];
    const node = nodes[nodes.length - 1];
    const fix = bbFixings(all, node);
    // Show the node's partial selection (not the incumbent) on the profile.
    packed.clear();
    for (const [i, v] of fix) (v ? packed : excluded).add(i);
    const order = (step.info.order as number[] | undefined) ?? [];
    if (node && node.status === 'branched' && order[node.depth] !== undefined) {
      cursor = order[node.depth];
      cursorLabel = `branch on x${sub(cursor)}`;
    }
  } else if (id === 'knapsack_dp') {
    cursor = (step.info.item as number | null | undefined) ?? null;
    if (cursor !== null) cursorLabel = `row ${step.k}: item ${cursor}`;
  }
  return { slot: run.sel.slot, packed, excluded, cursor, cursorLabel };
}

export function KnapsackStage({
  problem,
  runs,
  player,
  focus,
  focusIndex,
  reference,
  onCapacity,
  view,
}: KnapsackStageProps) {
  const colors = useChartColors();
  const layout = useRef<ItemsLayout | null>(null);
  const [dragC, setDragC] = useState<number | null>(null);
  const [hoverItem, setHoverItem] = useState<number | null>(null);
  const shown: KnapsackInstance = dragC === null ? problem : { ...problem, capacity: dragC };

  const focusTrace = useMemo(() => focus?.result.trace ?? [], [focus]);
  const focusK = Math.min(focusTrace.length - 1, Math.floor(player.localT(focusIndex) + 1e-9));
  const focusStep = focusTrace[Math.max(0, focusK)];
  const allNodes = useMemo(
    () => (focus?.sel.id === 'knapsack_branch_bound' ? bbNodesUpTo(focusTrace, focusK) : []),
    [focus, focusTrace, focusK],
  );

  const rows = runs
    .filter((r) => !r.error && r.result.trace.length)
    .map((r) => {
      const i = runs.indexOf(r);
      const tr = r.result.trace;
      const st = tr[Math.min(tr.length - 1, Math.floor(player.localT(i) + 1e-9))];
      return {
        label: SHORT[r.sel.id] ?? r.method.spec.name,
        slot: r.sel.slot,
        items: selectedItems(st?.x),
        value: st?.fun ?? null,
      };
    });
  const marks = focus && dragC === null ? focusMarks(focus, focusStep, allNodes) : null;

  const { canvasRef } = useCanvas((ctx, s) => {
    layout.current = drawItems(ctx, s.width, s.height, {
      problem: shown,
      colors,
      rows: dragC === null ? rows : [],
      focus: marks,
      capacityHot: dragC !== null,
      hoverItem,
    });
  });

  const local = (e: PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };
  const total = problem.weights.reduce((a, b) => a + b, 0);
  const clampC = (c: number) => Math.max(0, Math.min(total, Math.round(c)));
  const nearLine = (x: number, y: number) => {
    const L = layout.current;
    return !!L && Math.abs(x - L.xOf(problem.capacity)) < 9 && y > L.profileTop - 18;
  };
  const onKey = (e: KeyboardEvent<HTMLCanvasElement>) => {
    const d = { ArrowLeft: -1, ArrowRight: 1, ArrowDown: -1, ArrowUp: 1 }[e.key];
    if (d === undefined) return;
    e.preventDefault();
    e.stopPropagation();
    onCapacity(clampC(problem.capacity + d * (e.shiftKey ? 10 : 1)));
  };
  const hoverInfo =
    hoverItem !== null
      ? `item ${hoverItem}: v = ${problem.values[hoverItem]}, w = ${problem.weights[hoverItem]}, v/w = ${sig(problem.values[hoverItem] / problem.weights[hoverItem], 4)}`
      : null;

  const detail =
    focus && !focus.error && focus.sel.id !== 'knapsack_greedy' ? (
      focus.sel.id === 'knapsack_dp' ? (
        <DpTable
          problem={problem}
          trace={focusTrace}
          slot={focus.sel.slot}
          t={player.localT(focusIndex)}
          animate={!player.reducedMotion && player.playing}
        />
      ) : (
        <BbTree
          nodes={allNodes}
          order={(focusStep?.info.order as number[]) ?? []}
          current={allNodes.length ? allNodes[allNodes.length - 1].id : null}
          slot={focus.sel.slot}
          zStar={reference}
        />
      )
    ) : null;

  return (
    <div className={styles.knapsackStage} data-view={view} data-detail={detail ? '' : undefined}>
      {view !== 'detail' && (
        <div className={styles.itemsBox}>
          <canvas
            ref={canvasRef}
            className={styles.itemsCanvas}
            role="slider"
            tabIndex={0}
            data-own-keys=""
            aria-label="Knapsack capacity C (drag the line or use the arrow keys)"
            aria-valuemin={0}
            aria-valuemax={total}
            aria-valuenow={shown.capacity}
            aria-valuetext={`C = ${shown.capacity}; ${rows.map((r) => `${r.label}: z = ${r.value}`).join(', ')}`}
            data-hot={dragC !== null || undefined}
            onKeyDown={onKey}
            onPointerDown={(e) => {
              const [x, y] = local(e);
              if (!nearLine(x, y)) return;
              e.currentTarget.setPointerCapture(e.pointerId);
              setDragC(problem.capacity);
            }}
            onPointerMove={(e) => {
              const [x, y] = local(e);
              const L = layout.current;
              if (dragC !== null && L) {
                setDragC(clampC(L.wOf(x)));
                return;
              }
              e.currentTarget.style.cursor = nearLine(x, y) ? 'ew-resize' : '';
              const it = L ? itemAt(L, x, y) : null;
              if (it !== hoverItem) setHoverItem(it);
            }}
            onPointerUp={() => {
              if (dragC !== null) {
                onCapacity(dragC);
                setDragC(null);
              }
            }}
            onPointerCancel={() => setDragC(null)}
            onPointerLeave={() => setHoverItem(null)}
          />
          {hoverInfo && <p className={styles.hoverNote}>{hoverInfo}</p>}
        </div>
      )}
      {view !== 'items' && detail && <div className={styles.detailBox}>{detail}</div>}
    </div>
  );
}

function DpTable({
  problem,
  trace,
  slot,
  t,
  animate,
}: {
  problem: KnapsackInstance;
  trace: readonly Step[];
  slot: number;
  t: number;
  animate: boolean;
}) {
  const colors = useChartColors();
  const layout = useRef<DpLayout | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const k = Math.min(trace.length - 1, Math.floor(t + 1e-9));
  const frac = animate && k < trace.length - 1 ? t - k : 0;
  const C = problem.capacity;
  const n = problem.values.length;
  const fillTo = frac > 0 ? Math.floor(frac * (C + 1)) + 1 : 0;
  const path = useMemo(
    () => (k >= 1 ? dpBacktrackPath(trace, k, problem.weights, C) : []),
    [trace, k, problem.weights, C],
  );
  const { canvasRef } = useCanvas((ctx, s) => {
    layout.current = drawDpTable(ctx, s.width, s.height, {
      problem,
      trace,
      colors,
      slot,
      k,
      fillTo,
      path,
      hover,
    });
  });
  const row0 = (r: number) => trace[r]?.info.table_row as number[] | undefined;
  let note: ReactNode = null;
  if (hover) {
    const [row, d] = hover;
    const z = row0(row)?.[d];
    if (row > k || z === undefined) {
      note = (
        <>
          Row {row} is not filled yet (the playhead is at row {k}).
        </>
      );
    } else if (row === 0) {
      note = <Formula tex={`z_{0}(${d}) = 0`} />;
    } else {
      const w = problem.weights[row - 1],
        v = problem.values[row - 1];
      const above = row0(row - 1)?.[d];
      const take = (trace[row].info.take as boolean[] | undefined)?.[d];
      note =
        d < w ? (
          <>
            <Formula tex={`z_{${row}}(${d}) = z_{${row - 1}}(${d}) = ${z}`} /> (item {row - 1} does
            not fit: <Formula tex={`w_{${row - 1}} = ${w} > ${d}`} />)
          </>
        ) : (
          <>
            <Formula
              tex={`z_{${row}}(${d}) = \\max\\bigl(z_{${row - 1}}(${d}),\\; z_{${row - 1}}(${d} - ${w}) + ${v}\\bigr) = \\max(${above}, ${row0(row - 1)?.[d - w]} + ${v}) = ${z}`}
            />
            {take ? ` · item ${row - 1} packed` : ` · item ${row - 1} skipped`}
          </>
        );
    }
  }
  // Keyboard: arrow keys move a cell cursor (Shift: 10 capacities), Home/End jump in the row.
  const onKey = (e: KeyboardEvent<HTMLCanvasElement>) => {
    const [r, d] = hover ?? [Math.max(0, k), C];
    const big = e.shiftKey ? 10 : 1;
    const moves: Record<string, [number, number]> = {
      ArrowUp: [r - 1, d],
      ArrowDown: [r + 1, d],
      ArrowLeft: [r, d - big],
      ArrowRight: [r, d + big],
      Home: [r, 0],
      End: [r, C],
    };
    const next = moves[e.key];
    if (e.key === 'Escape') {
      if (hover) {
        setHover(null);
        e.stopPropagation();
      }
      return;
    }
    if (!next) return;
    e.preventDefault();
    e.stopPropagation();
    setHover([Math.max(0, Math.min(n, next[0])), Math.max(0, Math.min(C, next[1]))]);
  };
  return (
    <div className={styles.dpBox}>
      <canvas
        ref={canvasRef}
        className={styles.dpCanvas}
        tabIndex={0}
        data-own-keys={hover ? '' : undefined}
        role="img"
        aria-label={`Dynamic-programming table zⱼ(d): ${k} of ${trace.length - 1} item rows filled; z${sub(k)}(C) = ${trace[k]?.fun ?? '—'}. Use the arrow keys to read a cell.`}
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
        onPointerMove={(e) => {
          const L = layout.current;
          if (!L) return;
          const r = e.currentTarget.getBoundingClientRect();
          const cell = dpCellAt(L, e.clientX - r.left, e.clientY - r.top);
          if (cell?.join() !== hover?.join()) setHover(cell);
        }}
        onPointerLeave={() => setHover(null)}
      />
      <p className={styles.dpNote} aria-live="polite">
        {note ?? (
          <>
            Row <Formula tex="j" /> holds <Formula tex="z_j(d)" />, the best value from items 0 to{' '}
            <Formula tex="j-1" /> within capacity <Formula tex="d" />, and{' '}
            <Formula tex="z_j(d) = z_{j-1}(d)" /> if item <Formula tex="j-1" /> does not fit. The
            colored strip marks cells that pack item <Formula tex="j-1" />; the path backtracks the
            selection from <Formula tex={`(${k}, ${C})`} />. Hover a cell, or focus the table and
            use the arrow keys, to read its recurrence.
          </>
        )}
      </p>
    </div>
  );
}
