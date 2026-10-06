/**
 * "Try this": curated views that each reveal one phenomenon. A preset sets the problem, the
 * methods (with parameters), the shared degree and the focused method, and always drops edited
 * data (`d`), so the claim in its title is exactly what the viewer sees.
 */
import type { LabPreset, MethodSelection } from '../_shell';

export interface RegressionPreset extends LabPreset {
  /** Shared polynomial degree (URL `deg`). */
  degree?: number;
  /** Method shown in detail (URL `f`). */
  focus?: string;
  /** Playback speed multiplier the preset sets (default 1×): a 3-step exchange needs time. */
  speed?: number;
}

/** Default view: robust lines against least squares on the line with two gross outliers. */
export const DEFAULT_PROBLEM = 'outliers_linear';
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'linear_regression', slot: 0, params: {} },
  { id: 'huber_regression', slot: 1, params: {} },
  { id: 'theil_sen', slot: 2, params: {} },
];
export const DEFAULT_DEGREE = 2;

export const PRESETS: RegressionPreset[] = [
  {
    id: 'outliers',
    title: 'Two outliers tilt least squares',
    note: 'y₅ + 15 and y₁₆ − 20 pull the least-squares slope to 1.47. Huber and Theil–Sen stay near the true 2, yet least squares keeps the highest R².',
    problem: 'outliers_linear',
    methods: DEFAULT_SELECTION,
    focus: 'huber_regression',
  },
  {
    id: 'lad',
    title: 'IRLS for |r| crawls: 171 solves',
    note: 'Huber stops after 7 reweighted solves, LAD after 171: IRLS converges only linearly as the two contact residuals shrink toward 0 and their weights 1/|rᵢ| grow; the floor ε is reached only at k = 138.',
    problem: 'outliers_linear',
    methods: [
      { id: 'huber_regression', slot: 1, params: {} },
      { id: 'lad_regression', slot: 3, params: {} },
    ],
    focus: 'lad_regression',
  },
  {
    id: 'norms',
    title: 'ℓ∞ listens only to the extremes',
    note: 'Three exchanges, each swapping the worst point into the reference, level the minimax line on three points: it splits the two outliers, slope −0.11. Least squares tilts to 1.47; Theil–Sen stays at 2.01.',
    problem: 'outliers_linear',
    methods: [
      { id: 'linear_regression', slot: 0, params: {} },
      { id: 'chebyshev_minimax_line', slot: 2, params: {} },
      { id: 'theil_sen', slot: 3, params: {} },
    ],
    focus: 'chebyshev_minimax_line',
    speed: 0.5,
  },
  {
    id: 'overfit',
    title: 'Degree 12 chases the noise',
    note: 'The training error falls with every degree; the leave-one-out error is smallest at d = 3 and is 9× larger at d = 12. Ridge (λ = 0.1) damps the wiggles.',
    problem: 'noisy_quadratic',
    methods: [
      { id: 'polynomial_regression', slot: 0, params: {} },
      { id: 'ridge_regression', slot: 1, params: { lam: 0.1 } },
    ],
    degree: 12,
    focus: 'polynomial_regression',
  },
];
