/**
 * Combinatorial test problems — TS port of `numopt.problems.combinatorial` (kind "combinatorial").
 *
 * * `KnapsackInstance` — the 0-1 knapsack problem (Martello & Toth 1990, eq. 2.1–2.3):
 *   maximize Σᵢ vᵢxᵢ subject to Σᵢ wᵢxᵢ ≤ C, xᵢ ∈ {0, 1}.
 * * `TspInstance` — the symmetric Euclidean TSP on n cities with coordinates `coords` (n × 2).
 *
 * Seeded instances draw only from `Rng` (Mulberry32) in the order the Python builders document,
 * so the data are identical. Ids, names, optima and descriptions equal `problems.json`; `latex`
 * and `tags` are TS-only display fields (the export has no formula for this kind).
 */
import { Rng } from '../core/rng';
import { addProblem } from './registry';

export interface KnapsackInstance {
  kind: 'knapsack';
  id: string;
  name: string;
  values: number[];
  weights: number[];
  capacity: number;
  /** The optimal objective value z*, when known. */
  optimumValue: number | null;
  description: string;
  /** TS-only: the problem as a display formula (problem picker). */
  latex: string;
  tags: string[];
}

export interface TspInstance {
  kind: 'tsp';
  id: string;
  name: string;
  /** City coordinates, one (x, y) pair per city. */
  coords: [number, number][];
  /** Length of an optimal closed tour, when known. */
  optimumLength: number | null;
  description: string;
  latex: string;
  tags: string[];
}

export type CombinatorialInstance = KnapsackInstance | TspInstance;

export const KNAPSACK_LATEX =
  '\\begin{aligned} \\max_{\\mathbf{x} \\in \\{0,1\\}^n} \\;& \\textstyle\\sum_i v_i x_i \\\\ ' +
  '\\text{s.t.} \\;& \\textstyle\\sum_i w_i x_i \\le C \\end{aligned}';
/** The tour closes: index k + 1 is taken mod n, so the last term is the edge back to π₀. */
export const TSP_LATEX =
  '\\min_{\\pi} \\textstyle\\sum_{k=0}^{n-1} \\lVert \\mathbf{c}_{\\pi_k} - \\mathbf{c}_{\\pi_{(k+1) \\bmod n}} \\rVert_2';

export function isKnapsack(p: unknown): p is KnapsackInstance {
  return typeof p === 'object' && p !== null && (p as { kind?: string }).kind === 'knapsack';
}

export function isTsp(p: unknown): p is TspInstance {
  return typeof p === 'object' && p !== null && (p as { kind?: string }).kind === 'tsp';
}

/** Number of items (knapsack) or cities (TSP). */
export function instanceSize(p: CombinatorialInstance): number {
  return p.kind === 'knapsack' ? p.values.length : p.coords.length;
}

// ── Knapsack instances ──────────────────────────────────────────────────────────────────

/**
 * Weakly correlated items (Python `weakly_correlated_knapsack`). Draw order, for i = 0, …, n−1:
 * wᵢ = 10 + rng.integers(41), then vᵢ = wᵢ + rng.integers(21) − 5. Capacity C = ⌊Σwᵢ / 2⌋.
 */
export function weaklyCorrelatedKnapsack(
  n: number,
  seed: number,
): { values: number[]; weights: number[]; capacity: number } {
  const rng = new Rng(seed);
  const weights: number[] = [];
  const values: number[] = [];
  for (let i = 0; i < n; i++) {
    const w = 10 + rng.integers(41);
    const v = w + rng.integers(21) - 5;
    weights.push(w);
    values.push(v);
  }
  const total = weights.reduce((a, b) => a + b, 0);
  return { values, weights, capacity: Math.floor(total / 2) };
}

function knapsack(
  id: string,
  name: string,
  data: { values: number[]; weights: number[]; capacity: number },
  optimumValue: number,
  description: string,
  tags: string[],
): KnapsackInstance {
  return {
    kind: 'knapsack',
    id,
    name,
    ...data,
    optimumValue,
    description,
    latex: KNAPSACK_LATEX,
    tags: ['knapsack', ...tags],
  };
}

export const knapsack10 = addProblem(
  'combinatorial',
  knapsack(
    'knapsack_10',
    'Knapsack, 10 items',
    weaklyCorrelatedKnapsack(10, 7),
    173,
    '10 weakly correlated items (seed 7), capacity = half the total weight. ' +
      'Greedy by value/weight ratio finds 172; the optimum is 173.',
    ['seeded'],
  ),
);

export const knapsack20 = addProblem(
  'combinatorial',
  knapsack(
    'knapsack_20',
    'Knapsack, 20 items',
    weaklyCorrelatedKnapsack(20, 3),
    407,
    '20 weakly correlated items (seed 3), capacity = half the total weight. ' +
      'Greedy finds 397; the optimum is 407 (2²⁰ ≈ 10⁶ subsets for brute force).',
    ['seeded'],
  ),
);

export const knapsackGreedyTrap = addProblem(
  'combinatorial',
  knapsack(
    'knapsack_greedy_trap',
    'Greedy trap',
    { values: [3, 50, 50], weights: [1, 50, 50], capacity: 100 },
    100,
    'The small item has the best value/weight ratio, so greedy packs it first ' +
      'and then only one large item fits: greedy = 53, best single item = 50, optimum = 100. ' +
      "Even the 'best single item' fix stays near the ½-approximation bound.",
    ['greedy', 'worst case'],
  ),
);

// ── TSP instances ───────────────────────────────────────────────────────────────────────

/** City i gets `points[perm[i]]` with `perm = new Rng(seed).permutation(n)` (Python `_shuffle`). */
export function shufflePoints(points: [number, number][], seed: number): [number, number][] {
  const perm = new Rng(seed).permutation(points.length);
  return perm.map((p) => [points[p][0], points[p][1]]);
}

/**
 * The regular 12-gon of radius 40 centred at (50, 50), shuffled with seed 12.
 *
 * NOTE: the coordinates are the exact doubles Python computes with `math.cos`/`math.sin`
 * (glibc). V8's `Math.sin` differs from glibc in the last bit for some k (e.g. sin(4π/3)), so
 * the points are written out instead of recomputed; a unit test checks them against the
 * formula (within 1e-13) and against the seed-12 shuffle.
 */
export const CIRCLE_12_POINTS: [number, number][] = [
  [90.0, 50.0],
  [84.64101615137756, 70.0],
  [70.0, 84.64101615137754],
  [50.0, 90.0],
  [30.000000000000007, 84.64101615137756],
  [15.35898384862245, 70.0],
  [10.0, 50.00000000000001],
  [15.358983848622444, 30.00000000000001],
  [29.999999999999982, 15.358983848622465],
  [49.99999999999999, 10.0],
  [70.0, 15.358983848622458],
  [84.64101615137753, 29.999999999999982],
];

function tsp(
  id: string,
  name: string,
  coords: [number, number][],
  optimumLength: number,
  description: string,
  tags: string[],
): TspInstance {
  return {
    kind: 'tsp',
    id,
    name,
    coords,
    optimumLength,
    description,
    latex: TSP_LATEX,
    tags: ['tsp', ...tags],
  };
}

export const tspCircle12 = addProblem(
  'combinatorial',
  tsp(
    'tsp_circle_12',
    '12 cities on a circle',
    shufflePoints(CIRCLE_12_POINTS, 12),
    24.0 * 40.0 * Math.sin(Math.PI / 12),
    'Regular 12-gon of radius 40 centered at (50, 50), city order shuffled ' +
      '(seed 12). The optimum is the polygon perimeter 960·sin(π/12) ≈ 248.466.',
    ['convex position'],
  ),
);

export const tspGrid16 = addProblem(
  'combinatorial',
  tsp(
    'tsp_grid_16',
    '4 × 4 grid',
    shufflePoints(
      Array.from({ length: 16 }, (_, k) => [10.0 * (k % 4), 10.0 * Math.floor(k / 4)]),
      16,
    ),
    160.0,
    'A 4 × 4 grid with spacing 10, city order shuffled (seed 16). Every edge ' +
      'has length ≥ 10 and a boustrophedon cycle uses only length-10 edges: optimum = 160. ' +
      'Many optimal tours exist (ties everywhere).',
    ['ties'],
  ),
);

export const tspRandom15 = addProblem(
  'combinatorial',
  tsp(
    'tsp_random_15',
    '15 random cities',
    (() => {
      // Draw order: for each city, x = rng.integers(101), then y = rng.integers(101).
      const rng = new Rng(15);
      return Array.from({ length: 15 }, (): [number, number] => {
        const x = rng.integers(101);
        const y = rng.integers(101);
        return [x, y];
      });
    })(),
    399.1256152125949,
    '15 cities with integer coordinates drawn uniformly from {0, …, 100}² ' +
      '(seed 15). Optimum ≈ 399.126.',
    ['seeded'],
  ),
);

const CLUSTER_CENTRES: [number, number][] = [
  [20, 25],
  [75, 20],
  [30, 75],
  [80, 70],
];

export const tspCities20 = addProblem(
  'combinatorial',
  tsp(
    'tsp_cities_20',
    '20 clustered cities',
    (() => {
      // City i belongs to cluster i mod 4: x = cx + rng.integers(21) − 10, then y likewise.
      const rng = new Rng(21);
      return Array.from({ length: 20 }, (_, i): [number, number] => {
        const [cx, cy] = CLUSTER_CENTRES[i % 4];
        const x = cx + rng.integers(21) - 10;
        const y = cy + rng.integers(21) - 10;
        return [x, y];
      });
    })(),
    280.94443474166826,
    'Four clusters of five cities (integer offsets in [−10, 10] around fixed ' +
      'centers, seed 21). Good tours visit each cluster once. Optimum ≈ 280.944.',
    ['seeded', 'clusters'],
  ),
);
