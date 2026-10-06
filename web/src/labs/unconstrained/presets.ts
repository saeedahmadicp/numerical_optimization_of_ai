/**
 * Defaults and "Try this" presets of the unconstrained lab. Every number in a title or note is
 * checked against the run it sets in presets.test.ts.
 */
import type { LabPreset, MethodSelection } from '../_shell';

export const DEFAULT_PROBLEM = 'rosenbrock';

/**
 * First view: three ways to use a model of f on the Rosenbrock valley from 𝐱₀ = (−1.2, 1).
 * BFGS (38 iterations) learns the curvature from gradients, the dogleg trust region (24) trusts
 * its exact quadratic model only inside a disk, and Nelder–Mead (110) uses no derivatives at all.
 * All three finish within the ~7 s of the first playback, so each geometry is seen moving.
 */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'bfgs', slot: 0, params: {} },
  { id: 'trust_region_dogleg', slot: 1, params: {} },
  { id: 'nelder_mead', slot: 2, params: {} },
];

/** URL keys of the lab (besides p, m, x0). */
export const KEYS = {
  focus: 'f',
  view: 'v',
  metric: 'y',
  lens: 'lens',
} as const;

/** The worst start for steepest descent on quadratic_ill: 𝐱₀ ∝ κ𝐪₁ + 𝐪₂ (κ = 50). */
export const ZIGZAG_X0: [number, number] = [2.364, 1.848];

export const PRESETS: LabPreset[] = [
  {
    id: 'maximizer',
    title: 'Newton’s method converges to a maximizer',
    note:
      'From (0, 0) each Newton step goes to the model’s stationary point: a maximum, f = 181.6. ' +
      'Damped Newton and the trust region reach (3, 2).',
    problem: 'himmelblau',
    methods: [
      { id: 'pure_newton', slot: 0, params: {} },
      { id: 'damped_newton', slot: 1, params: {} },
      { id: 'trust_region_exact', slot: 2, params: {} },
    ],
    extra: { [KEYS.focus]: 'pure_newton' },
  },
  {
    id: 'zigzag',
    title: 'Steepest descent zigzags; CG needs two steps',
    note:
      'κ = 50, worst start: each exact step is orthogonal to the last and f shrinks by ' +
      '((κ − 1)/(κ + 1))² = 0.923, 382 times. CG ends in n = 2.',
    problem: 'quadratic_ill',
    methods: [
      { id: 'gradient_descent', slot: 0, params: { step_rule: 'exact_quadratic' } },
      { id: 'cg_fletcher_reeves', slot: 1, params: {} },
    ],
    start: ZIGZAG_X0,
    extra: { [KEYS.focus]: 'gradient_descent' },
  },
  {
    id: 'offview',
    title: 'Newton leaves the view; the trust region does not',
    note:
      'Newton’s 𝐱₂ = (0.76, −3.18) is below the view. The dogleg toward it is rejected ' +
      '(ρ = −0.41), Δ shrinks to ¼, and 24 steps follow the valley.',
    problem: 'rosenbrock',
    methods: [
      { id: 'pure_newton', slot: 0, params: {} },
      { id: 'trust_region_dogleg', slot: 1, params: {} },
    ],
    extra: { [KEYS.focus]: 'trust_region_dogleg' },
  },
  {
    id: 'schedules',
    title: 'Long steps beat 1/L; a restart beats both',
    note:
      'κ = 50. Gradient descent with α = 1/L takes 639 steps; the κ-aware silver schedule, with ' +
      'steps as long as 22.9/L, takes 239; FISTA with gradient restart takes 92.',
    problem: 'quadratic_ill',
    methods: [
      { id: 'gradient_descent', slot: 0, params: { step_rule: 'fixed', lr: 0.02 } },
      { id: 'silver_gd_strongly_convex', slot: 1, params: {} },
      { id: 'fista', slot: 2, params: {} },
    ],
    extra: { [KEYS.focus]: 'silver_gd_strongly_convex' },
  },
  {
    id: 'aa-saddle',
    title: 'Anderson acceleration stops at a saddle point',
    note:
      'AA solves ∇f = 0 with no descent test: from (1, 1) it stops at the saddle (0.087, 2.884) ' +
      'after 16 steps. Gradient descent (21) and ARC (10) reach the minimizer (3, 2).',
    problem: 'himmelblau',
    methods: [
      { id: 'anderson_gd', slot: 0, params: { lr: 0.01 } },
      { id: 'gradient_descent', slot: 1, params: {} },
      { id: 'arc', slot: 2, params: {} },
    ],
    start: [1, 1],
    extra: { [KEYS.focus]: 'anderson_gd' },
  },
  {
    id: 'arc-escape',
    title: 'Near a saddle point, ARC escapes',
    note:
      'Start next to the saddle (0, 0): pure Newton stops on it after 2 steps, super-universal ' +
      'regularized Newton after 3. ARC follows the negative curvature to (−0.090, 0.713) in 10.',
    problem: 'six_hump_camel',
    methods: [
      { id: 'pure_newton', slot: 0, params: {} },
      { id: 'reg_newton', slot: 1, params: { variant: 'super_universal' } },
      { id: 'arc', slot: 2, params: {} },
    ],
    start: [0.05, 0.05],
    extra: { [KEYS.focus]: 'arc' },
  },
  {
    id: 'adamw',
    title: 'Weight decay moves AdamW’s fixed point',
    note:
      'Adam reaches ‖∇f‖₂ ≤ 10⁻⁶ in 281 steps. AdamW’s decay −λ𝐱 balances its step near 𝐱⋆: ' +
      '‖∇f‖₂ stays at 1.7×10⁻³ for 5,000 steps.',
    problem: 'quadratic_bowl',
    methods: [
      { id: 'adam', slot: 0, params: {} },
      { id: 'adamw', slot: 1, params: {} },
    ],
    extra: { [KEYS.focus]: 'adam' },
  },
];
