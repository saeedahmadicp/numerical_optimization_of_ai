/**
 * Canvas drawing for the knapsack stage. Pure functions (no React).
 *
 * Items profile: items in ratio order as bars of width wᵢ and height vᵢ/wᵢ (so each bar's area
 * is vᵢ) along the cumulative-weight axis. The area left of the capacity line x = C is exactly
 * Dantzig's LP bound (the split item is cut by the line). Below it, one knapsack per method:
 * the current selection packed along [0, C] on the same weight scale.
 *
 * DP table: z_j(d) for rows j = 0..n and capacities d = 0..C as a sequential field, the
 * take_j(d) cells as a strip in the method color, the row being filled cell by cell with the
 * two cells it reads, and the backtrack path of the current selection.
 */
import type { Step } from '../../core/types';
import type { ChartColors } from '../../ui/colors';
import { SEQUENTIAL, sample } from '../../ui/colors';
import { sig } from '../../core/format';
import { drawMath, linearTicks, mathMain, mathSub, mathVar } from '../../viz';
import { labelFont, tickFont } from '../../viz/axes';

import type { KnapsackInstance } from '../../problems/combinatorial';
import { dantzigBound, ratioOrder } from '../../methods/combinatorial/knapsack';

export const rgba = (color: string, a: number) => {
  const m = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (!m) return color;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};

// ── Items profile + knapsacks ──────────────────────────────────────────────────────────

export interface KnapsackRow {
  label: string;
  slot: number;
  /** Selected items at the playhead. */
  items: number[];
  value: number | null;
}

export interface FocusMarks {
  slot: number;
  /** Items packed in the focus method's current state. */
  packed: Set<number>;
  /** Items examined and left out (greedy no-fit, B&B x = 0 on the current path). */
  excluded: Set<number>;
  /** The item this step looks at (cursor). */
  cursor: number | null;
  cursorLabel?: string;
}

export interface ItemsLayout {
  left: number;
  right: number;
  profileTop: number;
  profileBottom: number;
  rowsTop: number;
  rowH: number;
  /** Weight → px. */
  xOf: (w: number) => number;
  wOf: (px: number) => number;
  /** Items in ratio order with their cumulative start weight. */
  bars: { item: number; start: number; w: number; ratio: number }[];
  maxRatio: number;
  yOf: (r: number) => number;
}

export function itemsLayout(
  p: KnapsackInstance,
  width: number,
  height: number,
  nRows: number,
): ItemsLayout {
  // Room on the left for the row labels ("Greedy") at the label size.
  const left = 66,
    right = 14;
  const rowH = 22;
  const rowsH = nRows ? nRows * rowH + 22 : 0;
  const profileTop = 30,
    profileBottom = height - rowsH - 30;
  const order = ratioOrder(p.values, p.weights);
  let cum = 0;
  const bars = order.map((i) => {
    const b = { item: i, start: cum, w: p.weights[i], ratio: p.values[i] / p.weights[i] };
    cum += p.weights[i];
    return b;
  });
  const maxW = Math.max(cum, p.capacity, 1);
  const maxRatio = Math.max(...bars.map((b) => b.ratio), 1e-9);
  const span = width - left - right;
  return {
    left,
    right: width - right,
    profileTop,
    profileBottom,
    rowsTop: height - rowsH + 6,
    rowH,
    xOf: (w) => left + (w / maxW) * span,
    wOf: (px) => ((px - left) / span) * maxW,
    bars,
    maxRatio,
    yOf: (r) => profileBottom - (r / (maxRatio * 1.12)) * (profileBottom - profileTop),
  };
}

export function drawItems(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  o: {
    problem: KnapsackInstance;
    colors: ChartColors;
    rows: KnapsackRow[];
    focus: FocusMarks | null;
    /** Capacity line being dragged (px position), or hovered. */
    capacityHot?: boolean;
    hoverItem?: number | null;
  },
): ItemsLayout {
  const { problem: p, colors } = o;
  const L = itemsLayout(p, width, height, o.rows.length);
  const C = p.capacity;
  const xC = L.xOf(C);
  const base = L.profileBottom;

  // Axes: ratio ticks, weight ticks.
  ctx.save();
  ctx.font = tickFont(colors);
  ctx.fillStyle = colors.tick;
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const r of linearTicks(0, L.maxRatio * 1.12, 4)) {
    const y = Math.round(L.yOf(r)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(L.left, y);
    ctx.lineTo(L.right, y);
    ctx.stroke();
    ctx.fillText(sig(r, 3), L.left - 6, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const maxW = L.wOf(L.right);
  for (const w of linearTicks(0, maxW, Math.max(3, Math.floor((L.right - L.left) / 70)))) {
    ctx.fillText(String(w), L.xOf(w), base + 4);
  }
  // Axis names: vᵢ/wᵢ (top left, typeset like the axis names of drawAxes) and the weight axis.
  drawMath(
    ctx,
    [mathVar('v'), mathSub('i', 'italic'), mathMain('/'), mathVar('w'), mathSub('i', 'italic')],
    6,
    L.profileTop - 12,
    { size: 13, color: colors.text2 },
  );
  ctx.font = labelFont(colors);
  ctx.fillStyle = colors.text3;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.strokeStyle = colors.halo;
  ctx.lineWidth = 4;
  ctx.lineJoin = 'round';
  ctx.strokeText('cumulative weight, items by vᵢ/wᵢ ↓', L.right, base + 30);
  ctx.fillText('cumulative weight, items by vᵢ/wᵢ ↓', L.right, base + 30);
  ctx.restore();

  // LP relaxation: the part of the profile left of C (area = Dantzig bound).
  const [lp] = dantzigBound(p.values, p.weights, ratioOrder(p.values, p.weights), 0, C);
  ctx.save();
  for (const b of L.bars) {
    const x0 = L.xOf(b.start),
      x1 = L.xOf(Math.min(b.start + b.w, C));
    if (x1 <= x0) continue;
    ctx.fillStyle = rgba(colors.mode === 'dark' ? '#f1f0eb' : '#141413', 0.07);
    ctx.fillRect(x0, L.yOf(b.ratio), x1 - x0, base - L.yOf(b.ratio));
  }
  ctx.restore();

  // Bars.
  const f = o.focus;
  const fc = f ? colors.series[f.slot] : colors.text;
  const indexBoxes: { x0: number; x1: number; y0: number; y1: number }[] = [];
  ctx.save();
  for (const b of L.bars) {
    const x0 = L.xOf(b.start),
      x1 = L.xOf(b.start + b.w),
      y = L.yOf(b.ratio);
    const packed = f?.packed.has(b.item);
    ctx.fillStyle = packed
      ? rgba(fc, 0.5)
      : rgba(colors.mode === 'dark' ? '#f1f0eb' : '#141413', 0.035);
    ctx.fillRect(x0, y, x1 - x0, base - y);
    ctx.strokeStyle = packed ? fc : colors.axis;
    ctx.lineWidth = packed ? 1.5 : 1;
    ctx.strokeRect(
      Math.round(x0) + 0.5,
      Math.round(y) + 0.5,
      Math.max(1, Math.round(x1 - x0) - 1),
      Math.round(base - y),
    );
    if (o.hoverItem === b.item) {
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1, y + 1, x1 - x0 - 2, base - y - 2);
    }
    // Item index on the bar (left out where it would touch the previous index).
    if (x1 - x0 >= 7) {
      ctx.font = tickFont(colors);
      const t = String(b.item);
      const tw = ctx.measureText(t).width;
      const r = {
        x0: (x0 + x1) / 2 - tw / 2 - 1,
        x1: (x0 + x1) / 2 + tw / 2 + 1,
        y0: y - 14,
        y1: y - 1,
      };
      if (!indexBoxes.some((q) => q.x0 < r.x1 && r.x0 < q.x1 && q.y0 < r.y1 && r.y0 < q.y1)) {
        indexBoxes.push(r);
        ctx.fillStyle = colors.text2;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(t, (x0 + x1) / 2, y - 2);
      }
    }
    if (f?.excluded.has(b.item)) {
      const cx = (x0 + x1) / 2,
        cy = y + Math.min(14, (base - y) / 2);
      ctx.strokeStyle = colors.text2;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(cx - 4, cy - 4);
      ctx.lineTo(cx + 4, cy + 4);
      ctx.moveTo(cx + 4, cy - 4);
      ctx.lineTo(cx - 4, cy + 4);
      ctx.stroke();
    }
    if (f && f.cursor === b.item) {
      const cx = (x0 + x1) / 2;
      ctx.fillStyle = colors.text;
      ctx.beginPath();
      ctx.moveTo(cx, y - 14);
      ctx.lineTo(cx - 5, y - 21);
      ctx.lineTo(cx + 5, y - 21);
      ctx.closePath();
      ctx.fill();
      if (f.cursorLabel) {
        // Above the marker (clear of the item numbers on the neighboring bars), kept inside.
        ctx.font = labelFont(colors);
        const tw = ctx.measureText(f.cursorLabel).width;
        const lx = Math.max(L.left, Math.min(L.right - tw, cx - tw / 2));
        const ly = Math.max(L.profileTop - 18, y - 27);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.strokeStyle = colors.halo;
        ctx.lineWidth = 4;
        ctx.lineJoin = 'round';
        ctx.strokeText(f.cursorLabel, lx, ly);
        ctx.fillStyle = colors.text;
        ctx.fillText(f.cursorLabel, lx, ly);
      }
    }
  }
  ctx.restore();

  // Capacity line with its handle and the LP bound.
  ctx.save();
  ctx.strokeStyle = colors.text;
  ctx.lineWidth = o.capacityHot ? 2.4 : 1.6;
  // The line skips the axis-label band between the profile and the knapsacks.
  ctx.beginPath();
  ctx.moveTo(Math.round(xC) + 0.5, L.profileTop - 6);
  ctx.lineTo(Math.round(xC) + 0.5, base);
  if (o.rows.length) {
    ctx.moveTo(Math.round(xC) + 0.5, L.rowsTop - 3);
    ctx.lineTo(Math.round(xC) + 0.5, L.rowsTop + o.rows.length * L.rowH);
  }
  ctx.stroke();
  ctx.fillStyle = colors.text;
  ctx.beginPath();
  ctx.roundRect(xC - 5, L.profileTop - 14, 10, 14, 3);
  ctx.fill();
  ctx.fillStyle = colors.surface;
  ctx.fillRect(xC - 2, L.profileTop - 10, 1, 6);
  ctx.fillRect(xC + 1, L.profileTop - 10, 1, 6);
  // Labels right of C, or left of it when they would leave the plot (measured).
  const lpText = `shaded area = LP bound ${sig(lp, 5)}`;
  ctx.font = labelFont(colors);
  const lpW = ctx.measureText(lpText).width;
  const right = xC + 10 + lpW > width - 4;
  ctx.textAlign = right ? 'right' : 'left';
  ctx.textBaseline = 'middle';
  const tx = xC + (right ? -10 : 10);
  ctx.strokeStyle = colors.halo;
  ctx.lineWidth = 4;
  ctx.lineJoin = 'round';
  ctx.font = labelFont(colors);
  ctx.strokeText(`C = ${C}`, tx, L.profileTop - 7);
  ctx.fillStyle = colors.text;
  ctx.fillText(`C = ${C}`, tx, L.profileTop - 7);
  ctx.font = labelFont(colors);
  ctx.strokeText(lpText, tx, L.profileTop + 8);
  ctx.fillStyle = colors.text2;
  ctx.fillText(lpText, tx, L.profileTop + 8);
  ctx.restore();

  // Knapsacks (one row per method), on the same weight scale.
  ctx.save();
  o.rows.forEach((r, ri) => {
    const y = L.rowsTop + ri * L.rowH;
    const h = L.rowH - 7;
    const color = colors.series[r.slot] ?? colors.text;
    ctx.fillStyle = color;
    ctx.fillRect(6, y + h / 2 - 4, 8, 8);
    ctx.font = labelFont(colors);
    ctx.fillStyle = colors.text2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(r.label, 18, y + h / 2);
    // Track [0, C].
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(L.xOf(0)) + 0.5, Math.round(y) + 0.5, Math.round(xC - L.xOf(0)), h);
    let cum = 0;
    r.items.forEach((i, s) => {
      const x0 = L.xOf(cum),
        x1 = L.xOf(cum + p.weights[i]);
      cum += p.weights[i];
      ctx.fillStyle = rgba(color, s % 2 ? 0.62 : 0.88);
      ctx.fillRect(x0 + 0.5, y + 1, Math.max(1, x1 - x0 - 1), h - 1);
      ctx.font = tickFont(colors);
      if (x1 - x0 >= ctx.measureText(String(i)).width + 6) {
        ctx.fillStyle = colors.mode === 'dark' ? '#0d0d0c' : '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(String(i), (x0 + x1) / 2, y + h / 2 + 0.5);
      }
    });
    // z and W after the knapsack.
    ctx.font = tickFont(colors);
    ctx.fillStyle = colors.text;
    const label = r.value === null ? '' : `z = ${r.value}  W = ${cum}`;
    const after = xC + 8;
    if (after + ctx.measureText(label).width < width - 4) {
      ctx.textAlign = 'left';
      ctx.fillText(label, after, y + h / 2);
    } else {
      // No room right of C: a surface chip inside the track, so the text never sits on the bars.
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = colors.surface;
      ctx.beginPath();
      ctx.roundRect(xC - 10 - tw, y + 1.5, tw + 8, h - 2, 3);
      ctx.fill();
      ctx.textAlign = 'right';
      ctx.fillStyle = colors.text;
      ctx.fillText(label, xC - 6, y + h / 2);
    }
  });
  ctx.restore();
  return L;
}

/** The item under a pointer position in the profile, or null. */
export function itemAt(L: ItemsLayout, x: number, y: number): number | null {
  if (y < L.profileTop - 4 || y > L.profileBottom) return null;
  const w = L.wOf(x);
  const b = L.bars.find((b) => w >= b.start && w < b.start + b.w);
  return b && y >= L.yOf(b.ratio) - 14 ? b.item : null;
}

// ── DP table ───────────────────────────────────────────────────────────────────────────

export interface DpLayout {
  left: number;
  top: number;
  cellW: number;
  cellH: number;
  rows: number;
  cols: number;
}

export function dpLayout(n: number, C: number, width: number, height: number): DpLayout {
  const left = 74,
    top = 26,
    right = 10,
    bottom = 8;
  const cols = C + 1,
    rows = n + 1;
  return {
    left,
    top,
    cellW: (width - left - right) / cols,
    cellH: Math.min(40, (height - top - bottom) / rows),
    rows,
    cols,
  };
}

export function dpCellAt(L: DpLayout, x: number, y: number): [number, number] | null {
  const d = Math.floor((x - L.left) / L.cellW),
    row = Math.floor((y - L.top) / L.cellH);
  if (d < 0 || d >= L.cols || row < 0 || row >= L.rows) return null;
  return [row, d];
}

export function drawDpTable(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  o: {
    problem: KnapsackInstance;
    trace: readonly Step[];
    colors: ChartColors;
    slot: number;
    /** Rows 0..k complete; row k+1 filled up to `fillTo` columns (exclusive). */
    k: number;
    fillTo: number;
    /** Backtrack path cells (row, d) of the current selection. */
    path: [number, number][];
    hover?: [number, number] | null;
  },
): DpLayout {
  const { problem: p, trace, colors } = o;
  const n = p.values.length,
    C = p.capacity;
  const L = dpLayout(n, C, width, height);
  const zMax = Math.max(1, ...((trace[trace.length - 1]?.info.table_row as number[]) ?? [1]));
  const color = colors.series[o.slot] ?? colors.text;
  const dark = colors.mode === 'dark';
  // Sequential field, blended 30 % towards the surface so the take strips and the path lead.
  const bg = dark ? [20, 20, 19] : [252, 252, 251];
  const shade = (z: number) => {
    const t = z / zMax;
    const c = sample(SEQUENTIAL, dark ? 0.08 + 0.7 * t : 0.9 - 0.6 * t);
    const mix = (i: number) => Math.round(0.7 * c[i] + 0.3 * bg[i]);
    return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
  };
  const x = (d: number) => L.left + d * L.cellW;
  const y = (row: number) => L.top + row * L.cellH;

  // Column ticks (capacity d).
  ctx.save();
  ctx.font = tickFont(colors);
  ctx.fillStyle = colors.tick;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  let lastRight = -Infinity;
  for (const d of linearTicks(0, C, Math.max(3, Math.floor((width - L.left) / 60)))) {
    if (!Number.isInteger(d)) continue;
    // A tick that would touch the previous one (the wide "d = 0" first) is left out.
    const t = d === 0 ? 'd = 0' : String(d);
    const tw = ctx.measureText(t).width;
    const x0 = d === 0 ? x(0) : x(d) + L.cellW / 2 - tw / 2;
    if (x0 < lastRight + 6) continue;
    ctx.textAlign = 'left';
    ctx.fillText(t, x0, L.top - 4);
    lastRight = x0 + tw;
  }
  ctx.fillStyle = colors.text3;
  ctx.textAlign = 'left';
  ctx.fillText('row', 4, L.top - 4);
  ctx.textAlign = 'right';
  ctx.fillText('w/v', L.left - 6, L.top - 4);
  ctx.restore();

  // Cells.
  const fillRow = o.k + 1;
  for (let row = 0; row < L.rows; row++) {
    const filled = row <= o.k;
    const partial = row === fillRow;
    if (!filled && !partial) continue;
    const tr = trace[row];
    if (!tr) continue;
    const z = tr.info.table_row as number[];
    const take = tr.info.take as boolean[];
    const upto = filled ? L.cols : Math.max(0, Math.min(L.cols, o.fillTo));
    // Merge runs of equal z for speed.
    let d0 = 0;
    for (let d = 1; d <= upto; d++) {
      if (d === upto || z[d] !== z[d0]) {
        ctx.fillStyle = shade(z[d0]);
        ctx.fillRect(x(d0), y(row), x(d) - x(d0) + 0.3, L.cellH - 1);
        d0 = d;
      }
    }
    // take_j(d): a strip along the top of the cell.
    ctx.fillStyle = rgba(color, 0.9);
    let t0 = -1;
    for (let d = 0; d <= upto; d++) {
      const on = d < upto && take[d];
      if (on && t0 < 0) t0 = d;
      if (!on && t0 >= 0) {
        ctx.fillRect(x(t0), y(row), x(d) - x(t0), Math.max(2, L.cellH * 0.22));
        t0 = -1;
      }
    }
    // Numbers when cells are wide enough.
    // Numbers when cells are wide and tall enough for tick-size figures (else the shade carries z).
    if (L.cellW >= 24 && L.cellH >= 16) {
      ctx.font = tickFont(colors);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let d = 0; d < upto; d++) {
        ctx.fillStyle = dark
          ? z[d] / zMax > 0.55
            ? '#141413'
            : '#f1f0eb'
          : z[d] / zMax > 0.55
            ? '#fcfcfb'
            : '#141413';
        ctx.fillText(String(z[d]), x(d) + L.cellW / 2, y(row) + L.cellH / 2 + 0.5);
      }
    }
  }
  // Empty future rows: a hairline frame.
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (let row = 0; row < L.rows; row++) {
    if (row <= o.k) continue;
    ctx.strokeRect(L.left + 0.5, y(row) + 0.5, L.cols * L.cellW - 1, L.cellH - 2);
  }

  // Row labels: z_j and the item of the row.
  // Short rows label every `every`-th row (and the current one) so labels never touch.
  ctx.font = tickFont(colors);
  ctx.textBaseline = 'middle';
  const every = Math.max(1, Math.ceil(13 / L.cellH));
  for (let row = 0; row < L.rows; row++) {
    const cy = y(row) + L.cellH / 2;
    if (row !== o.k && (row % every !== 0 || Math.abs(row - o.k) < every)) continue;
    ctx.fillStyle = row <= o.k ? colors.text2 : colors.text3;
    ctx.textAlign = 'left';
    ctx.fillText(row === 0 ? 'z₀' : `z${sub(row)}`, 6, cy);
    if (row > 0 && L.cellH >= 11) {
      ctx.fillStyle = colors.text3;
      ctx.textAlign = 'right';
      ctx.fillText(`${p.weights[row - 1]}/${p.values[row - 1]}`, L.left - 6, cy);
    }
  }

  // The cell being written and the two cells it reads.
  if (fillRow < L.rows && o.fillTo > 0 && o.fillTo <= L.cols) {
    const d = o.fillTo - 1;
    const w = p.weights[fillRow - 1];
    const cx = x(d) + L.cellW / 2,
      cy = y(fillRow) + L.cellH / 2;
    ctx.save();
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x(d) - 1, y(fillRow) - 1, L.cellW + 2, L.cellH + 1);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = colors.text2;
    const arrow = (sx: number, sy: number) => {
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(cx, cy - L.cellH / 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(sx, sy, 2.4, 0, Math.PI * 2);
      ctx.fillStyle = colors.text;
      ctx.fill();
    };
    arrow(cx, y(fillRow - 1) + L.cellH / 2);
    if (d - w >= 0) arrow(x(d - w) + L.cellW / 2, y(fillRow - 1) + L.cellH / 2);
    ctx.restore();
  }

  // Backtrack path of the current selection.
  if (o.path.length > 1) {
    const pts = o.path.map(([row, d]) => [x(d) + L.cellW / 2, y(row) + L.cellH / 2] as const);
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const [w, c] of [
      [5, colors.halo],
      [2, color],
    ] as const) {
      ctx.beginPath();
      pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.strokeStyle = c;
      ctx.lineWidth = w;
      ctx.stroke();
    }
    // Diagonal steps = items packed: a dot where the path leaves the row.
    for (let s = 0; s < o.path.length - 1; s++) {
      if (o.path[s][1] !== o.path[s + 1][1]) {
        ctx.beginPath();
        ctx.arc(pts[s][0], pts[s][1], 3.6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = colors.halo;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  if (o.hover) {
    const [row, d] = o.hover;
    ctx.strokeStyle = colors.accent;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x(d) - 0.5, y(row) - 0.5, L.cellW + 1, L.cellH);
  }
  return L;
}

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (n: number) => [...String(n)].map((c) => SUBS[Number(c)]).join('');
