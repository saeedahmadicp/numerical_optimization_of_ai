/**
 * Constrained lab: defaults, "Try this" presets and URL codecs (module constants, so their
 * identity is stable across renders).
 *
 * Every number quoted in a preset note is the Python reference's result for that exact setup
 * (checked in geometry.test.ts against the TS port, which matches the fixtures).
 */
import { codecs } from '../../app/useUrlState';
import type { LabPreset, MethodSelection } from '../_shell';

export const DEFAULT_PROBLEM = 'rosenbrock_unit_disk';

/**
 * The first view: three ways to reach the same boundary point. The barrier stays strictly
 * inside the disk, the quadratic penalty approaches from outside (infeasible by O(1/μ)), and
 * SQP leaves the disk once and returns in 10 iterations.
 */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'sqp', slot: 0, params: {} },
  { id: 'log_barrier', slot: 1, params: {} },
  { id: 'quadratic_penalty', slot: 2, params: {} },
];

/**
 * Lab-specific first views where the problem's domain is far larger than the geometry: hs21
 * lives on [2, 50] × [−50, 50] (the bounds of Hock & Schittkowski), but x⋆ = (2, 0), the default
 * start (10, 10) and every path stay in a few units of the corner x = 2, y = 10x − 10.
 */
export const VIEW_DOMAIN: Record<string, [[number, number], [number, number]]> = {
  hs21: [
    [0, 13],
    [-4, 12],
  ],
};

export type Landscape = 'f' | 'merit';
export type Metric = 'kkt' | 'viol' | 'gap';

export const KEYS = { landscape: 'v', metric: 'y', kkt: 'kkt' } as const;
export const LANDSCAPE_CODEC = codecs.oneOf<Landscape>(['f', 'merit']);
export const METRIC_CODEC = codecs.oneOf<Metric>(['kkt', 'viol', 'gap']);
export const BOOL_CODEC = codecs.bool;

export const PRESETS: LabPreset[] = [
  {
    id: 'central-path',
    title: 'The barrier rides the central path',
    note: 'Each stage minimizes f − (1/t)Σ log(−cᵢ); as t grows ×4 the log wall flattens and x⋆(t) slides to the circle.',
    problem: 'quadratic_disk',
    methods: [{ id: 'log_barrier', slot: 1, params: { t0: 0.1, mu: 4 } }],
    extra: { [KEYS.landscape]: 'merit', [KEYS.metric]: 'kkt' },
  },
  {
    id: 'outside-in',
    title: 'Penalty stays outside; multipliers fix it',
    note: 'Each penalty minimizer violates the constraint by λ⋆/μ (1.2×10⁻⁷ at μ = 10⁷); the augmented Lagrangian keeps μ = 10 and updates λ instead.',
    problem: 'quadratic_disk',
    methods: [
      { id: 'quadratic_penalty', slot: 2, params: {} },
      { id: 'augmented_lagrangian', slot: 3, params: {} },
    ],
    extra: { [KEYS.metric]: 'viol', [KEYS.landscape]: 'merit' },
  },
  {
    id: 'fw-sublinear',
    title: 'Frank–Wolfe crawls at O(1/k)',
    note: 'With γₖ = 2/(k+2) it needs 625 iterations where projected gradient needs 11, on the same disk.',
    problem: 'quadratic_disk',
    methods: [
      { id: 'frank_wolfe', slot: 0, params: {} },
      { id: 'projected_gradient', slot: 1, params: {} },
    ],
    extra: { [KEYS.landscape]: 'f', [KEYS.metric]: 'kkt' },
  },
  {
    id: 'kkt-circle',
    title: '−∇f lines up with λ∇c at x⋆',
    note: 'On x² + y² = 1 the KKT arrows close only at x⋆, where λ⋆ = 1/√2 (the equality multiplier); SQP leaves the view once on the way.',
    problem: 'circle_eq',
    methods: [
      { id: 'sqp', slot: 0, params: {} },
      { id: 'augmented_lagrangian', slot: 3, params: {} },
    ],
    extra: { [KEYS.landscape]: 'f', [KEYS.metric]: 'kkt' },
  },
];
