/**
 * TSP stage: one panel per compared method (small multiples on the same cities), each drawing
 * the tour of its current step and the geometry of the move that produced it. Cities can be
 * dragged (the instance is edited and every method re-runs); a click on a city sets the
 * nearest-neighbor start. Keyboard: focus the stage, [ and ] pick a city, arrow keys move it
 * (Shift ×5), Enter makes it the nearest-neighbor start, Esc lets go.
 */
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { TracePlayer } from '../../play/useTracePlayer';
import { useCanvas, useElementSize } from '../../viz';
import { useChartColors } from '../../ui/theme';
import { Swatch } from '../../ui/components';
import { int, sig } from '../../core/format';
import type { Step } from '../../core/types';
import type { TspInstance } from '../../problems/combinatorial';
import type { LabRun } from '../_shell';
import {
  chooseGrid,
  cityDomain,
  fmtGap,
  gapPercent,
  shownLength,
  shortMethodName,
  unit,
} from './model';
import { drawTspPanel, type Mapping, type Pt } from './tspDraw';
import styles from './CombinatorialLab.module.css';

export interface TspStageProps {
  problem: TspInstance;
  /** The panels to draw (on phones, only the run in focus). */
  runs: readonly LabRun[];
  /** Index of a run in the player's trace list. */
  indexOf: (run: LabRun) => number;
  player: TracePlayer;
  reference: number | null;
  /** The reference is the best tour found (the optimum of an edited instance is unknown). */
  bestFound: boolean;
  /** Commit a moved city (instance coordinates). */
  onMoveCity: (i: number, x: number, y: number) => void;
  /** Nearest-neighbor start city (null when the method is not selected). */
  nnStart: number | null;
  onPickStart?: (i: number) => void;
}

interface Drag {
  i: number;
  x: number;
  y: number;
  moved: boolean;
  panel: number;
  px: number;
  py: number;
}

export function TspStage({
  problem,
  runs,
  indexOf,
  player,
  reference,
  bestFound,
  onMoveCity,
  nnStart,
  onPickStart,
}: TspStageProps) {
  const box = useRef<HTMLDivElement>(null);
  const size = useElementSize(box);
  const { cols, rows } = chooseGrid(Math.max(1, runs.length), size.width, size.height);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [hot, setHot] = useState<number | null>(null);
  const domain = useMemo(() => cityDomain(problem.coords), [problem.coords]);
  const coords: Pt[] = useMemo(
    () =>
      problem.coords.map((c, i) =>
        drag && drag.i === i ? ([drag.x, drag.y] as Pt) : ([c[0], c[1]] as Pt),
      ),
    [problem.coords, drag],
  );

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const n = problem.coords.length;
    if (e.key === ']' || e.key === '[') {
      e.preventDefault();
      setHot((h) => (h === null ? 0 : (h + (e.key === ']' ? 1 : n - 1)) % n));
      return;
    }
    if (hot === null) return;
    if (e.key === 'Escape') {
      setHot(null);
      e.stopPropagation();
      return;
    }
    if (e.key === 'Enter' && onPickStart) {
      onPickStart(hot);
      e.preventDefault();
      return;
    }
    const span = domain[0][1] - domain[0][0];
    const stepSize = (e.shiftKey ? 5 : 1) * Math.max(0.1, Math.round(span / 100));
    const delta: Record<string, [number, number]> = {
      ArrowLeft: [-stepSize, 0],
      ArrowRight: [stepSize, 0],
      ArrowUp: [0, stepSize],
      ArrowDown: [0, -stepSize],
    };
    const dxy = delta[e.key];
    if (!dxy) return;
    e.preventDefault();
    e.stopPropagation();
    const [x, y] = problem.coords[hot];
    onMoveCity(hot, x + dxy[0], y + dxy[1]);
  };

  return (
    <div
      ref={box}
      className={styles.tspGrid}
      style={{
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
      }}
      role="group"
      aria-label={`${problem.name}: ${problem.coords.length} cities. Press [ or ] to pick a city, arrow keys to move it${onPickStart ? ', Enter to start the nearest-neighbor tour there' : ''}.`}
      tabIndex={0}
      data-plot-focus=""
      data-own-keys={hot !== null ? '' : undefined}
      onKeyDown={onKeyDown}
      onBlur={() => setHot(null)}
    >
      {runs.map((r) => (
        <TspPanel
          key={r.sel.id}
          run={r}
          index={indexOf(r)}
          coords={coords}
          domain={domain}
          player={player}
          reference={reference}
          bestFound={bestFound}
          hot={drag?.i ?? hot}
          nnStart={nnStart}
          drag={drag}
          setDrag={setDrag}
          onCommit={(d) => {
            if (d.moved) onMoveCity(d.i, d.x, d.y);
            else if (onPickStart) onPickStart(d.i);
          }}
        />
      ))}
    </div>
  );
}

function TspPanel({
  run,
  index,
  coords,
  domain,
  player,
  reference,
  bestFound,
  hot,
  nnStart,
  drag,
  setDrag,
  onCommit,
}: {
  run: LabRun;
  index: number;
  coords: Pt[];
  domain: [[number, number], [number, number]];
  player: TracePlayer;
  reference: number | null;
  bestFound: boolean;
  hot: number | null;
  nnStart: number | null;
  drag: Drag | null;
  setDrag: (d: Drag | null) => void;
  onCommit: (d: Drag) => void;
}) {
  const colors = useChartColors();
  const mapping = useRef<Mapping | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const trace = run.result.trace;
  const id = run.sel.id;
  const t = player.localT(index);
  const k = Math.min(trace.length - 1, Math.floor(t + 1e-9));
  const frac = player.reducedMotion || !player.playing ? 0 : Math.max(0, Math.min(1, t - k));
  const step = trace[Math.max(0, k)];
  const tour = (step?.info.tour as number[] | undefined) ?? [];

  const schedule = useMemo(() => {
    if (id !== 'tsp_simulated_annealing' || !trace.length) return null;
    const T0 = trace[0].info.temperature as number;
    const params = run.sel.params;
    const t0 = Number(params.t0 ?? 1),
      tMin = Number(params.t_min ?? 1e-3);
    return { t0: T0, tMin: (T0 / t0) * tMin };
  }, [id, trace, run.sel.params]);

  const { canvasRef } = useCanvas((ctx, s) => {
    if (!trace.length || coords.length === 0) return;
    mapping.current = drawTspPanel(ctx, s.width, s.height, {
      coords,
      domain,
      colors,
      slot: run.sel.slot,
      methodId: id,
      trace,
      k: Math.max(0, k),
      frac,
      hot: hot ?? hover,
      startCity: id === 'tsp_nearest_neighbor' ? nnStart : null,
      schedule,
      labels: coords.length <= 24 && s.width >= 220,
      reference,
      bestFound,
    });
  });

  const cityAt = (e: PointerEvent<HTMLCanvasElement>): number | null => {
    const m = mapping.current;
    if (!m) return null;
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left,
      py = e.clientY - r.top;
    let best: number | null = null,
      bd = 13 * 13;
    coords.forEach((c, i) => {
      const [x, y] = m.toPx(c);
      const dd = (x - px) ** 2 + (y - py) ** 2;
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    });
    return best;
  };

  const gap = step ? gapPercent('tsp', step, reference) : null;
  const n = run.result.nIter;
  const readout = useMemo(() => tspReadout(id, trace, n, reference), [id, trace, n, reference]);
  const heldKarpWord = (s: Step | undefined) =>
    s?.info.closed ? 'tour closed' : `layer |S| = ${s?.info.subset_size ?? 0}`;

  return (
    <figure className={styles.panel} aria-label={run.method.spec.name}>
      <figcaption className={styles.panelHead}>
        <Swatch slot={run.sel.slot} size={9} />
        <span className={styles.panelName} title={run.method.spec.name}>
          <span className={styles.nameFull}>{run.method.spec.name}</span>
          <span className={styles.nameShort}>{shortMethodName(run.method.spec.name)}</span>
        </span>
        {step && (
          // Every number sits in a box as wide as its widest value over the run (tabular
          // figures, right-aligned), so the right-aligned readout never moves during playback.
          <span className={styles.panelNums}>
            {id === 'tsp_held_karp' ? (
              <span>
                <span className={styles.numBox} style={{ minWidth: `${readout.wWord}ch` }}>
                  {heldKarpWord(step)}
                </span>
              </span>
            ) : (
              <span>
                {unit(id, 1)}{' '}
                <span
                  className={styles.numBox}
                  data-align="right"
                  style={{ minWidth: `${readout.wK}ch` }}
                >
                  {int(step.k)}
                </span>{' '}
                of {int(n)}
              </span>
            )}
            <span>
              <i>L</i>
              {id === 'tsp_ant_colony' ? (
                // L_best from the first iteration on; kept in the layout at k = 0 (hidden).
                <sub className={styles.subBest} data-hidden={step.k === 0 || undefined}>
                  best
                </sub>
              ) : null}{' '}
              ={' '}
              <span
                className={styles.numBox}
                data-align="right"
                style={{ minWidth: `${readout.wL}ch` }}
              >
                {readout.fmtL(shownLength(step))}
              </span>
            </span>
            <span
              title={
                bestFound
                  ? 'Relative gap to the best tour found by these runs'
                  : 'Relative gap to the optimum'
              }
            >
              {bestFound ? 'gap to best' : 'gap'}{' '}
              <span
                className={styles.numBox}
                data-align="right"
                style={{ minWidth: `${readout.wGap}ch` }}
              >
                {fmtGap(gap)}
              </span>
            </span>
          </span>
        )}
      </figcaption>
      {run.error ? (
        <p className={styles.panelError}>
          {run.error}. {id === 'tsp_held_karp' ? 'Pick an instance with at most 16 cities.' : ''}
        </p>
      ) : (
        <canvas
          ref={canvasRef}
          className={styles.panelCanvas}
          role="img"
          aria-label={
            step
              ? `${run.method.spec.name}, ${id === 'tsp_held_karp' ? heldKarpWord(step) : `${unit(id, 1)} ${int(step.k)} of ${int(n)}`}: ${step.info.closed === false ? 'path' : 'tour'} ${tour.join(' → ')}, length ${sig(step.fun ?? NaN, 5)}.`
              : run.method.spec.name
          }
          data-dragging={drag?.panel === index || undefined}
          data-hover={hover !== null || undefined}
          onPointerDown={(e) => {
            const c = cityAt(e);
            if (c === null) return;
            e.currentTarget.setPointerCapture(e.pointerId);
            const [x, y] = coords[c];
            setDrag({ i: c, x, y, moved: false, panel: index, px: e.clientX, py: e.clientY });
          }}
          onPointerMove={(e) => {
            if (drag && drag.panel === index && mapping.current) {
              const r = e.currentTarget.getBoundingClientRect();
              const moved = drag.moved || Math.hypot(e.clientX - drag.px, e.clientY - drag.py) > 3;
              const [x, y] = mapping.current.fromPx(e.clientX - r.left, e.clientY - r.top);
              const [[x0, x1], [y0, y1]] = domain;
              setDrag({
                ...drag,
                moved,
                x: Math.min(x1, Math.max(x0, x)),
                y: Math.min(y1, Math.max(y0, y)),
              });
              return;
            }
            const c = cityAt(e);
            if (c !== hover) setHover(c);
          }}
          onPointerUp={() => {
            if (drag && drag.panel === index) {
              onCommit(drag);
              setDrag(null);
            }
          }}
          onPointerCancel={() => setDrag(null)}
          onPointerLeave={() => setHover(null)}
        />
      )}
    </figure>
  );
}

/**
 * The panel readout's run-wide format: L with a fixed number of decimals (five significant
 * digits at the run's largest length: 298.99, 283.00), and the width in ch of the widest step
 * count, length, gap and Held–Karp layer word over the whole trace.
 */
function tspReadout(
  id: string,
  trace: readonly Step[],
  n: number,
  reference: number | null,
): { fmtL: (v: number) => string; wK: number; wL: number; wGap: number; wWord: number } {
  let maxL = 0;
  for (const s of trace) {
    const L = Math.abs(shownLength(s));
    if (Number.isFinite(L)) maxL = Math.max(maxL, L);
  }
  const intDigits = maxL >= 1 ? Math.floor(Math.log10(maxL)) + 1 : 1;
  const decimals = Math.max(0, 5 - intDigits);
  const fmtL = (v: number) => (Number.isFinite(v) ? v.toFixed(decimals) : '—');
  let wK = int(n).length,
    wL = 1,
    wGap = 1,
    wWord = 1;
  for (const s of trace) {
    wK = Math.max(wK, int(s.k).length);
    wL = Math.max(wL, fmtL(shownLength(s)).length);
    wGap = Math.max(wGap, fmtGap(gapPercent('tsp', s, reference)).length);
    if (id === 'tsp_held_karp')
      wWord = Math.max(
        wWord,
        (s.info.closed ? 'tour closed' : `layer |S| = ${s.info.subset_size ?? 0}`).length,
      );
  }
  return { fmtL, wK, wL, wGap, wWord };
}
