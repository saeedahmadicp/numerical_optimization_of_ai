import { describe, expect, it } from 'vitest';
import { chipLine2, layoutOffView, sideOfPx, type OffViewRun, type Rect } from './offview';

const W = 800,
  H = 600;
// Data = pixels, y down (the layout only needs toPx).
const toPx = (x: number, y: number): [number, number] => [x, y];
const measure = () => 150;
const overlap = (a: Rect, b: Rect) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

function run(points: [number, number][], muted = false): OffViewRun {
  return { points, reached: points.length - 1, name: 'Newton', color: '#000', muted };
}

describe('off-view chips', () => {
  it('sides', () => {
    expect(sideOfPx(10, 700, W, H)).toBe('below');
    expect(sideOfPx(10, -5, W, H)).toBe('above');
    expect(sideOfPx(-5, 10, W, H)).toBe('left');
    expect(sideOfPx(900, 10, W, H)).toBe('right');
    expect(sideOfPx(10, 10, W, H)).toBeNull();
  });

  it('one chip per run and edge, naming the latest iterate and the count', () => {
    const pts: [number, number][] = [[400, 300]];
    for (let k = 1; k <= 9; k++) pts.push([100 + 60 * k, 700 + k]);
    const chips = layoutOffView([run(pts)], toPx, W, H, [], measure);
    expect(chips).toHaveLength(1);
    expect(chips[0]).toMatchObject({ side: 'below', count: 9, k: 9 });
    expect(chipLine2('Newton', 9, 'below')).toBe('Newton · 9 iterates below');
    expect(chipLine2('Newton', 1, 'below')).toBe('Newton · below the view');
    // Clear of the x tick labels (bottom 24 px) and the y tick labels (left 40 px).
    expect(chips[0].box.y + chips[0].box.h).toBeLessThanOrEqual(H - 24);
    expect(chips[0].box.x).toBeGreaterThanOrEqual(40);
  });

  it('chips of two runs leaving at the same place do not overlap, nor hit the obstacles', () => {
    const a = run([
      [400, 300],
      [400, 900],
    ]);
    const b = run(
      [
        [400, 300],
        [410, 900],
      ],
      true,
    );
    const c = run([
      [60, 300],
      [60, -900],
    ]);
    const legend: Rect = { x: 44, y: 10, w: 300, h: 80 };
    const chips = layoutOffView([a, b, c], toPx, W, H, [legend], measure);
    expect(chips).toHaveLength(3);
    for (let i = 0; i < chips.length; i++) {
      expect(overlap(chips[i].box, legend)).toBe(false);
      for (let j = i + 1; j < chips.length; j++)
        expect(overlap(chips[i].box, chips[j].box)).toBe(false);
    }
    // The focused (unmuted) run keeps the spot beside its exit (not over the dashed step).
    expect(chips[0].box.x).toBe(414);
  });

  it('only iterates the playhead has reached', () => {
    const r = {
      ...run([
        [400, 300],
        [400, 350],
        [400, 900],
      ]),
      reached: 1,
    };
    expect(layoutOffView([r], toPx, W, H, [], measure)).toHaveLength(0);
  });
});

describe('α as a power of two', async () => {
  const { powerOfTwo } = await import('./format');
  it('formats 2⁻ⁱ exactly and anything else with 3 digits', () => {
    expect(powerOfTwo(1)).toBe('1');
    expect(powerOfTwo(0.5)).toBe('2⁻¹');
    expect(powerOfTwo(2 ** -27)).toBe('2⁻²⁷');
    expect(powerOfTwo(0.3)).toBe('0.3');
  });
});
