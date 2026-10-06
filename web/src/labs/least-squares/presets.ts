/**
 * The least-squares lab's first view and its "Try this" presets. Every number a note states is
 * checked against the Python reference in tests/least-squares/lab.test.ts (fixture cases
 * `presets` from gen_least_squares_fixture.py).
 */
import type { LabPreset, MethodSelection } from '../_shell';

/**
 * The first view: the exponential fit from (0.2, 2.8), a nearly flat start. Gauss–Newton's
 * first full steps overshoot and Armijo backtracking halves them; Levenberg–Marquardt rejects
 * three trials, so μ rises before it falls — both reach the same fit.
 */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'gauss_newton', slot: 0, params: {} },
  { id: 'levenberg_marquardt', slot: 1, params: {} },
];
export const DEFAULT_PROBLEM = 'exp_decay_fit';
/** Lab starts that differ from the problem's registered x0 (stated in the Python call). */
export const LAB_START: Record<string, [number, number]> = { exp_decay_fit: [0.2, 2.8] };

export const PRESETS: LabPreset[] = [
  {
    id: 'newton',
    title: 'Gauss–Newton is Newton on Rosenbrock',
    note: 'The Jacobian is square and invertible, so every full step is a Newton step on the two residual equations: the first leaves the view at (1, −3.84), the second lands on the solution (1, 1).',
    problem: 'rosenbrock_ls',
    start: [-1.2, 1],
    methods: [
      { id: 'gauss_newton', slot: 0, params: { line_search: 'none' } },
      { id: 'levenberg_marquardt', slot: 1, params: {} },
    ],
  },
  {
    id: 'rank',
    title: 'A full step can kill the amplitude',
    note: 'From (0.5, 2.8) the first undamped step overshoots to (2.45, −9.04); the second cancels the amplitude to rounding level, where even its sign is noise. The Jacobian loses rank and Gauss–Newton stops. The damped steps keep the fit alive.',
    problem: 'exp_decay_fit',
    start: [0.5, 2.8],
    methods: [
      { id: 'gauss_newton', slot: 0, params: { line_search: 'none' } },
      { id: 'levenberg_marquardt', slot: 1, params: {} },
    ],
  },
  {
    id: 'mirror',
    title: 'A short arc fits two centers',
    note: 'From (1.2, 3.5) both methods converge to the mirror center (1.30, 2.74): a local minimum whose objective, 1.19, is two hundred times the global one.',
    problem: 'circle_fit',
    start: [1.2, 3.5],
    methods: [
      { id: 'gauss_newton', slot: 0, params: {} },
      { id: 'levenberg_marquardt', slot: 1, params: {} },
    ],
  },
  {
    id: 'scale',
    title: 'Damping ignores the scale of the parameters',
    note: 'The rate V is near 200 and the constant K near 0.06, yet the damping shortens both by the same amount: Levenberg–Marquardt started with heavy damping needs 27 iterations where Gauss–Newton needs 7.',
    problem: 'michaelis_menten',
    start: [205, 0.08],
    methods: [
      { id: 'gauss_newton', slot: 0, params: {} },
      { id: 'levenberg_marquardt', slot: 1, params: { tau: 1 } },
    ],
  },
];
