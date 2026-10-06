/**
 * Small static illustrations for the lab gallery — one per lab id, drawn on a canvas in the
 * chart tokens (ink + at most two method colors), so the gallery reads as one system.
 */
import type { ChartColors } from '../ui/colors';
import { Rng } from '../core/rng';

type Draw = (ctx: CanvasRenderingContext2D, w: number, h: number, c: ChartColors) => void;

const TAU = Math.PI * 2;

function curve(
  ctx: CanvasRenderingContext2D,
  f: (x: number) => number,
  x0: number,
  x1: number,
  map: (x: number, y: number) => [number, number],
  n = 120,
) {
  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    const x = x0 + ((x1 - x0) * i) / n;
    const [px, py] = map(x, f(x));
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
}

function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  halo: string,
  r = 3.2,
) {
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, r + 1.6, 0, TAU);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, TAU);
  ctx.fill();
}

function ink(ctx: CanvasRenderingContext2D, c: ChartColors, alpha = 0.75, width = 1.5) {
  ctx.strokeStyle = c.text;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
}

function axisLine(ctx: CanvasRenderingContext2D, c: ChartColors, y: number, w: number) {
  ctx.save();
  ctx.strokeStyle = c.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(12, y);
  ctx.lineTo(w - 12, y);
  ctx.stroke();
  ctx.restore();
}

/** Map unit box [0,1]² to the canvas with padding (y up). */
const box =
  (w: number, h: number, pad = 16) =>
  (x: number, y: number): [number, number] => [
    pad + x * (w - 2 * pad),
    h - pad - y * (h - 2 * pad),
  ];

function ellipses(
  ctx: CanvasRenderingContext2D,
  c: ChartColors,
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  rot: number,
  n = 6,
) {
  ctx.save();
  ctx.strokeStyle = c.isoStrong;
  ctx.lineWidth = 1;
  for (let i = 1; i <= n; i++) {
    ctx.beginPath();
    ctx.ellipse(cx, cy, (rx * i) / n, (ry * i) / n, rot, 0, TAU);
    ctx.stroke();
  }
  ctx.restore();
}

function polyline(
  ctx: CanvasRenderingContext2D,
  pts: [number, number][],
  color: string,
  width = 1.75,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  ctx.restore();
}

const DRAW: Record<string, Draw> = {
  roots(ctx, w, h, c) {
    const m = box(w, h);
    const f = (x: number) => 0.5 + 0.9 * (x - 0.55) ** 3 + 0.25 * (x - 0.55);
    axisLine(ctx, c, m(0, 0.5)[1], w);
    ctx.fillStyle = c.series[0];
    ctx.globalAlpha = 0.1;
    const [a] = m(0.3, 0),
      [b] = m(0.75, 0);
    ctx.fillRect(a, 10, b - a, h - 20);
    ctx.globalAlpha = 1;
    ink(ctx, c);
    curve(ctx, f, 0, 1, m);
    ctx.stroke();
    ctx.globalAlpha = 1;
    const x1 = 0.9,
      s = 2.7 * (x1 - 0.55) ** 2 + 0.25;
    ctx.strokeStyle = c.series[1];
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(...m(0.62, f(x1) + s * (0.62 - x1)));
    ctx.lineTo(...m(1, f(x1) + s * (1 - x1)));
    ctx.stroke();
    ctx.setLineDash([]);
    dot(ctx, ...m(x1, f(x1)), c.series[1], c.halo);
    dot(ctx, ...m(0.55, 0.5), c.text, c.halo, 2.6);
  },
  systems(ctx, w, h, c) {
    const m = box(w, h);
    ink(ctx, c, 0.6);
    curve(ctx, (x) => 0.15 + 0.7 * x * x, 0, 1, m);
    ctx.stroke();
    curve(ctx, (x) => 0.95 - 0.8 * x, 0, 1, m);
    ctx.stroke();
    ctx.globalAlpha = 1;
    const pts: [number, number][] = [m(0.15, 0.3), m(0.42, 0.52), m(0.56, 0.5), m(0.585, 0.482)];
    polyline(ctx, pts, c.series[0]);
    pts.forEach((p) => dot(ctx, ...p, c.series[0], c.halo, 2.5));
  },
  linalg(ctx, w, h, c) {
    const n = 5,
      size = Math.min((w - 40) / n, (h - 28) / n),
      x0 = (w - size * n) / 2,
      y0 = (h - size * n) / 2;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const zero = j < i && i > 0 && j < 2;
        ctx.fillStyle = i === 1 && j === 1 ? c.series[0] : zero ? c.grid : c.text;
        ctx.globalAlpha = i === 1 && j === 1 ? 0.9 : zero ? 1 : i === 1 || j === 1 ? 0.28 : 0.12;
        ctx.beginPath();
        ctx.roundRect(x0 + j * size + 2, y0 + i * size + 2, size - 4, size - 4, 3);
        ctx.fill();
      }
    ctx.globalAlpha = 1;
  },
  scalar(ctx, w, h, c) {
    const m = box(w, h, 18);
    const f = (x: number) => 0.18 + 2.2 * (x - 0.58) ** 2;
    ink(ctx, c);
    curve(ctx, f, 0, 1, m);
    ctx.stroke();
    ctx.globalAlpha = 1;
    const brackets = [
      [0.05, 0.95],
      [0.25, 0.95],
      [0.4, 0.82],
      [0.47, 0.7],
    ];
    brackets.forEach(([a, b], i) => {
      const y = h - 10 - i * 5;
      ctx.strokeStyle = c.series[0];
      ctx.globalAlpha = 0.35 + i * 0.2;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(m(a, 0)[0], y);
      ctx.lineTo(m(b, 0)[0], y);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
    dot(ctx, ...m(0.58, f(0.58)), c.series[0], c.halo);
  },
  'line-search'(ctx, w, h, c) {
    const m = box(w, h);
    const phi = (a: number) => 0.8 - 1.6 * a + 2.1 * a * a;
    ink(ctx, c);
    curve(ctx, phi, 0, 1, m);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = c.series[1];
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(...m(0, 0.8));
    ctx.lineTo(...m(1, 0.8 - 0.55));
    ctx.stroke();
    ctx.setLineDash([]);
    [1, 0.5, 0.25].forEach((a, i) =>
      dot(ctx, ...m(a, phi(a)), i === 1 ? c.series[0] : c.tick, c.halo, 2.8),
    );
  },
  unconstrained(ctx, w, h, c) {
    const cx = w * 0.62,
      cy = h * 0.55;
    ellipses(ctx, c, cx, cy, w * 0.5, h * 0.3, -0.35, 7);
    const zig: [number, number][] = [];
    for (let i = 0; i < 9; i++)
      zig.push([
        w * 0.14 + (i * (cx - w * 0.14)) / 8,
        cy - h * 0.28 + (i % 2 ? 22 : -4) * (1 - i / 9) + (i * h * 0.28) / 8,
      ]);
    zig.push([cx, cy]);
    polyline(ctx, zig, c.series[0], 1.5);
    const smooth: [number, number][] = [];
    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      smooth.push([
        w * 0.12 + t * (cx - w * 0.12) + Math.sin(t * 3.4) * 14 * (1 - t),
        h * 0.86 - t * (h * 0.86 - cy) - Math.sin(t * 2.6) * 8,
      ]);
    }
    polyline(ctx, smooth, c.series[1]);
    dot(ctx, cx, cy, c.text, c.halo, 2.6);
  },
  'least-squares'(ctx, w, h, c) {
    const m = box(w, h);
    const rng = new Rng(7);
    const f = (x: number) => 0.15 + 0.7 / (1 + Math.exp(-9 * (x - 0.5)));
    for (let i = 0; i < 14; i++) {
      const x = 0.04 + (i / 13) * 0.92,
        y = f(x) + (rng.random() - 0.5) * 0.2;
      ctx.strokeStyle = c.series[1];
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(...m(x, y));
      ctx.lineTo(...m(x, f(x)));
      ctx.stroke();
      ctx.globalAlpha = 1;
      dot(ctx, ...m(x, y), c.text, c.halo, 2.2);
    }
    ctx.strokeStyle = c.series[0];
    ctx.lineWidth = 1.75;
    curve(ctx, f, 0, 1, m);
    ctx.stroke();
  },
  global(ctx, w, h, c) {
    const m = box(w, h);
    const f = (x: number) =>
      0.5 +
      0.18 * Math.sin(18 * x) +
      0.25 * (x - 0.6) ** 2 * 4 -
      0.12 * Math.exp(-((x - 0.67) ** 2) / 0.004);
    ink(ctx, c);
    curve(ctx, f, 0, 1, m, 200);
    ctx.stroke();
    ctx.globalAlpha = 1;
    const rng = new Rng(3);
    for (let i = 0; i < 7; i++) {
      const x = rng.random();
      dot(ctx, ...m(x, f(x)), c.tick, c.halo, 2.2);
    }
    dot(ctx, ...m(0.67, f(0.67)), c.series[0], c.halo, 3.4);
  },
  stochastic(ctx, w, h, c) {
    const m = box(w, h);
    const rng = new Rng(11);
    for (const [slot, rate, noise] of [
      [0, 3.2, 0.09],
      [1, 5.5, 0.05],
    ] as const) {
      const pts: [number, number][] = [];
      for (let i = 0; i <= 60; i++) {
        const x = i / 60;
        pts.push(
          m(x, 0.1 + 0.8 * Math.exp(-rate * x) + (rng.random() - 0.5) * noise * (1 - x * 0.5)),
        );
      }
      polyline(ctx, pts, c.series[slot], 1.4);
    }
  },
  constrained(ctx, w, h, c) {
    const cx = w * 0.3,
      cy = h * 0.35;
    ellipses(ctx, c, cx, cy, w * 0.6, h * 0.5, 0.2, 7);
    ctx.fillStyle = c.series[0];
    ctx.globalAlpha = 0.08;
    ctx.beginPath();
    ctx.moveTo(w * 0.45, 0);
    ctx.lineTo(w, 0);
    ctx.lineTo(w, h);
    ctx.lineTo(w * 0.2, h);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = c.series[0];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(w * 0.45, 0);
    ctx.lineTo(w * 0.2, h);
    ctx.stroke();
    const pts: [number, number][] = [
      [w * 0.85, h * 0.8],
      [w * 0.6, h * 0.62],
      [w * 0.42, h * 0.5],
      [w * 0.385, h * 0.33],
    ];
    polyline(ctx, pts, c.series[1]);
    dot(ctx, w * 0.385, h * 0.33, c.series[1], c.halo);
  },
  lp(ctx, w, h, c) {
    const poly: [number, number][] = [
      [0.15, 0.15],
      [0.75, 0.15],
      [0.88, 0.45],
      [0.62, 0.85],
      [0.22, 0.7],
    ];
    const m = box(w, h);
    ctx.fillStyle = c.series[0];
    ctx.globalAlpha = 0.09;
    ctx.beginPath();
    poly.forEach(([x, y], i) => (i ? ctx.lineTo(...m(x, y)) : ctx.moveTo(...m(x, y))));
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 0.6;
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.globalAlpha = 1;
    polyline(ctx, [m(0.15, 0.15), m(0.75, 0.15), m(0.88, 0.45), m(0.62, 0.85)], c.series[1], 2);
    poly.forEach(([x, y]) => dot(ctx, ...m(x, y), c.tick, c.halo, 2.2));
    dot(ctx, ...m(0.62, 0.85), c.series[1], c.halo, 3.4);
  },
  combinatorial(ctx, w, h, c) {
    const rng = new Rng(5);
    const pts: [number, number][] = Array.from({ length: 9 }, () => [
      16 + rng.random() * (w - 32),
      14 + rng.random() * (h - 28),
    ]);
    const cx = w / 2,
      cy = h / 2;
    pts.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    polyline(ctx, [...pts, pts[0]], c.series[0], 1.5);
    pts.forEach((p) => dot(ctx, ...p, c.text, c.halo, 2.6));
  },
  integration(ctx, w, h, c) {
    const m = box(w, h);
    const f = (x: number) => 0.3 + 0.45 * Math.sin(2.6 * x + 0.4) ** 2 + 0.1 * x;
    const n = 6;
    for (let i = 0; i < n; i++) {
      const a = i / n,
        b = (i + 1) / n;
      ctx.fillStyle = c.series[0];
      ctx.globalAlpha = i % 2 ? 0.14 : 0.22;
      ctx.beginPath();
      ctx.moveTo(...m(a, 0));
      ctx.lineTo(...m(a, f(a)));
      ctx.lineTo(...m(b, f(b)));
      ctx.lineTo(...m(b, 0));
      ctx.closePath();
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ink(ctx, c);
    curve(ctx, f, 0, 1, m);
    ctx.stroke();
    ctx.globalAlpha = 1;
    axisLine(ctx, c, m(0, 0)[1], w);
  },
  differentiation(ctx, w, h, c) {
    const m = box(w, h);
    ctx.strokeStyle = c.series[0];
    ctx.lineWidth = 1.75;
    curve(
      ctx,
      (x) =>
        Math.log10(10 ** (3 * (x - 0.55)) * 3 + 10 ** (-2.2 * (x - 0.55)) * 0.6 + 0.4) / 2.2 + 0.25,
      0,
      1,
      m,
    );
    ctx.stroke();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = c.tick;
    ctx.lineWidth = 1;
    curve(ctx, (x) => 0.12 + x * 0.85, 0.05, 1, m);
    ctx.stroke();
    curve(ctx, (x) => 0.95 - x * 0.75, 0, 0.95, m);
    ctx.stroke();
    ctx.setLineDash([]);
  },
  interpolation(ctx, w, h, c) {
    const m = box(w, h);
    const runge = (x: number) => 1 / (1 + 25 * (2 * x - 1) ** 2);
    const xs = Array.from({ length: 9 }, (_, i) => i / 8);
    const lag = (x: number) =>
      xs.reduce(
        (s, xi, i) =>
          s + runge(xi) * xs.reduce((p, xj, j) => (j === i ? p : (p * (x - xj)) / (xi - xj)), 1),
        0,
      );
    const mm = (x: number, y: number) => m(x, 0.15 + y * 0.6);
    ctx.strokeStyle = c.series[1];
    ctx.lineWidth = 1.5;
    curve(ctx, (x) => Math.max(-0.35, Math.min(1.3, lag(x))), 0, 1, mm, 200);
    ctx.stroke();
    ink(ctx, c, 0.55, 1.25);
    curve(ctx, runge, 0, 1, mm);
    ctx.stroke();
    ctx.globalAlpha = 1;
    xs.forEach((x) => dot(ctx, ...mm(x, runge(x)), c.text, c.halo, 2.4));
  },
  regression(ctx, w, h, c) {
    const m = box(w, h);
    const rng = new Rng(19);
    for (let i = 0; i < 26; i++) {
      const x = rng.random(),
        y = 0.2 + 0.6 * x + rng.normal(0, 0.09);
      dot(ctx, ...m(x, y), c.tick, c.halo, 2);
    }
    ctx.strokeStyle = c.series[0];
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    ctx.moveTo(...m(0, 0.2));
    ctx.lineTo(...m(1, 0.8));
    ctx.stroke();
  },
};

export function drawThumbnail(
  id: string,
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  colors: ChartColors,
): void {
  ctx.save();
  (DRAW[id] ?? DRAW.unconstrained)(ctx, w, h, colors);
  ctx.restore();
}
