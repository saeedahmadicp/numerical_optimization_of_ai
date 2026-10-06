/**
 * Defaults and "Try this" presets of the global lab. Every claim in a title or note was checked
 * against the run it sets (seed included): see tests/global/lab.test.ts.
 */
import type { LabPreset, MethodSelection } from '../_shell';

/**
 * First view: three search strategies on Rastrigin from 𝐱₀ = (3.3, −2.6), seed 0. The swarm
 * reaches the origin; annealing (α = 0.98, about 460 steps) freezes in a side basin and CMA-ES
 * with its default λ = 6 settles on the ring of minima at f ≈ 1 — global search settling is
 * not global optimality.
 */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'simulated_annealing', slot: 0, params: { alpha: 0.98 } },
  { id: 'particle_swarm', slot: 1, params: {} },
  { id: 'cma_es', slot: 2, params: {} },
];

export const DEFAULT_PROBLEM = 'rastrigin';

/** URL key of the shared random seed. */
export const SEED_KEY = 's';

export const PRESETS: LabPreset[] = [
  {
    id: 'side-basins',
    title: 'Small populations stop in side basins',
    note:
      'On Rastrigin, CMA-ES ($\\lambda = 6$) and greedy DE/best/1 contract on the ring of minima ' +
      'at $f \\approx 1$. The 20-particle swarm reaches the origin.',
    problem: 'rastrigin',
    methods: [
      { id: 'cma_es', slot: 0, params: {} },
      { id: 'differential_evolution', slot: 1, params: { strategy: 'best/1/bin' } },
      { id: 'particle_swarm', slot: 2, params: {} },
    ],
    extra: { [SEED_KEY]: '0' },
  },
  {
    id: 'rosenbrock-ellipse',
    title: 'CMA-ES learns the Rosenbrock valley',
    note:
      'The sampling ellipse turns and stretches along the curved valley, then shrinks onto ' +
      '(1, 1): the covariance learns the local shape of $f$.',
    problem: 'rosenbrock',
    methods: [{ id: 'cma_es', slot: 0, params: {} }],
    extra: { [SEED_KEY]: '1' },
  },
  {
    id: 'fast-cooling',
    title: 'Fast cooling freezes in a side basin',
    note:
      'With $\\alpha = 0.9$, $T_k$ falls tenfold every 22 steps and the walker freezes near ' +
      '(−1, −1), $f \\approx 2$. Basin hopping never cools and reaches the origin.',
    problem: 'rastrigin',
    methods: [
      { id: 'simulated_annealing', slot: 0, params: { alpha: 0.9 } },
      { id: 'basin_hopping', slot: 1, params: {} },
    ],
    extra: { [SEED_KEY]: '0' },
  },
  {
    id: 'log-cooling',
    title: 'Logarithmic cooling is still hot at k = 2,000',
    note:
      '$T_k = T_0 \\ln 2 / \\ln(k + 1)$ is the schedule of the convergence theorem (Hajek 1988). ' +
      'After the 2,000-step budget $T \\approx 0.91$, far above $T_{\\min} = 10^{-3}$.',
    problem: 'ackley',
    methods: [{ id: 'simulated_annealing', slot: 0, params: { cooling: 'logarithmic' } }],
    extra: { [SEED_KEY]: '0' },
  },
];
