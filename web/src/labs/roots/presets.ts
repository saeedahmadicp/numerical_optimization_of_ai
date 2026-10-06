/**
 * The roots lab's opening view and its "Try this" presets. Each preset sets the problem, the
 * methods, the start x₀ and the bracket (`br`), so a preset is also a shareable link.
 */
import type { LabPreset, MethodSelection } from '../_shell';

/**
 * Opening view: Wallis' cubic with three rates on one clock — bisection gains one bit per step
 * (33 steps), Brent's interpolation and Newton's tangents finish in 5.
 */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'bisection', slot: 0, params: {} },
  { id: 'newton', slot: 1, params: {} },
  { id: 'brent', slot: 2, params: {} },
];

export const PRESETS: LabPreset[] = [
  {
    id: 'cycle',
    title: 'Newton cycles on x³ − 2x + 2',
    note: 'From x₀ = 0 the tangents send 0 → 1 → 0 forever. Brent, bracketed on [−2, −1], finds the root.',
    problem: 'newton_cycle',
    methods: [
      { id: 'newton', slot: 0, params: {} },
      { id: 'brent', slot: 1, params: {} },
    ],
    start: 0,
    extra: { br: '-2,-1' },
  },
  {
    id: 'stall',
    title: 'Regula falsi stalls on x¹⁰ − 1',
    note: 'The steep right end never moves: 95 chords. Illinois halves the stuck end’s value and needs 14.',
    problem: 'x10_minus_1',
    methods: [
      { id: 'regula_falsi', slot: 0, params: {} },
      { id: 'illinois', slot: 1, params: {} },
    ],
    start: 1.3,
    extra: { br: '0,1.3' },
  },
  {
    id: 'diverge',
    title: 'Newton diverges on arctan x from x₀ = 1.5',
    note: 'Beyond |x₀| ≈ 1.3917 every tangent overshoots further; bisection on [−1.5, 2] cannot fail.',
    problem: 'atan_newton',
    methods: [
      { id: 'newton', slot: 0, params: {} },
      { id: 'bisection', slot: 1, params: {} },
    ],
    start: 1.5,
    extra: { br: '-1.5,2' },
  },
  {
    id: 'double',
    title: 'A double root slows Newton to rate ½',
    note: 'At x = 1 the tangent flattens with f: the error halves instead of squaring (Halley: rate ⅓).',
    problem: 'double_root',
    methods: [
      { id: 'newton', slot: 0, params: {} },
      { id: 'halley', slot: 1, params: {} },
    ],
    start: 2,
    extra: { br: '-3,0' },
  },
];
