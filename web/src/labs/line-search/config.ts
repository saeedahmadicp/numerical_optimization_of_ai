/**
 * Line-search lab constants: the default view, the "Try this" presets and the URL codecs.
 * (Kept out of the component files: those export components only.)
 */
import { codecs } from '../../app/useUrlState';
import type { LabPreset, MethodSelection } from '../_shell';
import type { WindowMode } from './geometry';

export type Direction = 'steepest' | 'newton';
export type Metric = 'gap' | 'slope';

/** URL keys of this lab (besides p, m, x0). */
export const KEYS = { direction: 'd', window: 'w', focus: 'f', metric: 'y' } as const;

export const DIRECTION_CODEC = codecs.oneOf<Direction>(['steepest', 'newton']);
export const WINDOW_CODEC = codecs.oneOf<WindowMode>(['near', 'all']);
export const METRIC_CODEC = codecs.oneOf<Metric>(['gap', 'slope']);

export const DEFAULT_PROBLEM = 'himmelblau';

/**
 * The first view: steepest descent from (0, 0) on Himmelblau, where φ falls faster than its
 * tangent. Backtracking halves α = 1 three times; strong Wolfe brackets [0, 1] and zooms by
 * quadratic interpolation; Goldstein bisects to α = 0.172, because its lower line excludes the
 * minimizer of φ (α ≈ 0.127). The conditions drawn are strong Wolfe's.
 */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'backtracking', slot: 0, params: {} },
  { id: 'strong_wolfe', slot: 1, params: {} },
  { id: 'goldstein', slot: 2, params: {} },
];
export const DEFAULT_FOCUS = 'strong_wolfe';

export const PRESETS: LabPreset[] = [
  {
    id: 'goldstein-excludes',
    title: 'Goldstein shuts out the minimizer of φ',
    note: 'On Himmelblau φ falls faster than its tangent, so the minimizer lies below the lower Goldstein line.',
    problem: 'himmelblau',
    start: [0, 0],
    methods: [
      { id: 'goldstein', slot: 0, params: {} },
      { id: 'strong_wolfe', slot: 1, params: { c2: 0.1 } },
    ],
    extra: { d: 'steepest', f: 'goldstein' },
  },
  {
    id: 'short-step',
    title: 'Backtracking never lengthens a short step',
    note: 'From α₀ = 10⁻³ Armijo holds at once. Strong Wolfe doubles α until φ rises, then one quadratic fit lands on the minimizer.',
    problem: 'quadratic_ill',
    start: [-2, 2],
    methods: [
      { id: 'backtracking', slot: 0, params: { alpha0: 0.001 } },
      { id: 'strong_wolfe', slot: 1, params: { alpha0: 0.001, c2: 0.1 } },
    ],
    extra: { d: 'steepest', f: 'strong_wolfe' },
  },
  {
    id: 'cubic-zoom',
    title: 'Strong Wolfe zooms by cubic interpolation',
    note: 'Eight doublings overshoot a dip of Ackley; two cubic fits land in it. Weak Wolfe accepts the overshoot, where φ′ > 0.',
    problem: 'ackley',
    start: [2.6, -3.4],
    methods: [
      { id: 'strong_wolfe', slot: 0, params: { alpha0: 0.001, c2: 0.1 } },
      { id: 'weak_wolfe', slot: 1, params: { alpha0: 0.001, c2: 0.1 } },
    ],
    extra: { d: 'steepest', f: 'strong_wolfe' },
  },
  {
    id: 'newton-unit',
    title: 'Newton’s step passes at α = 1',
    note: 'A well-scaled direction: the first trial already meets every condition.',
    problem: 'rosenbrock',
    start: [-1.2, 1],
    methods: [
      { id: 'backtracking', slot: 0, params: {} },
      { id: 'strong_wolfe', slot: 1, params: {} },
      { id: 'exact_quadratic', slot: 2, params: {} },
    ],
    extra: { d: 'newton', f: 'exact_quadratic' },
  },
];
