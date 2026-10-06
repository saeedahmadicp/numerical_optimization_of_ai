/**
 * Defaults and "Try this" presets of the differentiation lab. Every preset sets the problem, the
 * methods, the point x0 and the shared sweep (h0, K), so a preset is also a shareable link.
 */
import { param } from '../../core/registry';
import type { ParamSpec } from '../../core/types';
import type { LabPreset, MethodSelection } from '../_shell';

/** URL keys of the shared sweep and the view. */
export const KEYS = { h0: 'h0', levels: 'K', tol: 'tol', view: 'v' } as const;

export const DEFAULT_PROBLEM = 'exp_0_1';
export const DEFAULT_H0 = 0.1;
/** 40 halvings reach h = 9×10⁻¹⁴: deep enough that every V shows its round-off branch. */
export const DEFAULT_LEVELS = 40;
export const DEFAULT_TOL = 1e-6;

/**
 * The first view is the textbook figure: errors fall like h, h², h⁴ until cancellation takes
 * over (each V bottoms out near √ε, ε¹ᐟ³, ε¹ᐟ⁵), while the complex step never cancels.
 */
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'forward_difference', slot: 0, params: {} },
  { id: 'central_difference', slot: 1, params: {} },
  { id: 'five_point_stencil', slot: 2, params: {} },
  { id: 'complex_step', slot: 3, params: {} },
];

/** One sweep for every method (brand: a comparison uses one stopping test for every method). */
export const SWEEP_SPECS: ParamSpec[] = [
  param.float('h0', DEFAULT_H0, {
    min: 1e-12,
    max: 10,
    log: true,
    label: 'Largest step',
    tex: 'h_0',
    help: 'Level k uses hₖ = h₀/2ᵏ, for every method.',
  }),
  param.int('levels', DEFAULT_LEVELS, {
    min: 0,
    max: 60,
    label: 'Halvings',
    tex: 'K',
    help: 'The sweep runs k = 0, …, K; it never stops early, so the whole V is recorded.',
  }),
  param.float('tol', DEFAULT_TOL, {
    min: 1e-15,
    max: 1e-1,
    log: true,
    label: 'Tolerance',
    tex: '\\mathrm{tol}',
    help: "Converged when the selected level's error bound ≤ tol·max(1, |D|).",
  }),
];

const sweep = (h0: number, levels: number) => ({
  [KEYS.h0]: String(h0),
  [KEYS.levels]: String(levels),
});

export const PRESETS: LabPreset[] = [
  {
    id: 'v',
    title: 'Truncation meets round-off: the V',
    note: 'eˣ at x₀ = 1. Errors fall like h, h², h⁴ until cancellation wins near ε¹ᐟ², ε¹ᐟ³, ε¹ᐟ⁵. The complex step never cancels; until round-off it shares the central error |h²f‴/6|, so the two curves overlap.',
    problem: 'exp_0_1',
    start: 1,
    methods: DEFAULT_SELECTION,
    extra: sweep(0.1, 40),
  },
  {
    id: 'complex',
    title: 'Complex step: exact at h = 10⁻¹⁹',
    note: 'Once x₀ ± h round to x₀ the central difference returns 0. The complex step divides the imaginary part of f(x₀ + ih) by h, subtracts nothing and stays at ε.',
    problem: 'arctan_deriv',
    start: 0.5,
    methods: [
      { id: 'central_difference', slot: 1, params: {} },
      { id: 'complex_step', slot: 3, params: {} },
    ],
    extra: sweep(0.1, 60),
  },
  {
    id: 'kink',
    title: 'A kink fools Richardson',
    note: '|x − 0.3| at x₀ = 0.35: stencils wider than 0.05 straddle the kink. The central difference is exact from h = 0.05; extrapolation drags the bad rows along.',
    problem: 'abs_kink',
    start: 0.35,
    methods: [
      { id: 'central_difference', slot: 1, params: {} },
      { id: 'richardson_extrapolation', slot: 0, params: {} },
    ],
    extra: sweep(0.4, 20),
  },
  {
    id: 'second',
    title: 'f″ loses twice as fast',
    note: 'Dividing by h² instead of h: the round-off branch climbs like ε/h², so the second difference keeps only about half the digits.',
    problem: 'gaussian',
    start: 0.5,
    methods: [
      { id: 'central_difference', slot: 1, params: {} },
      { id: 'second_derivative_central', slot: 2, params: {} },
    ],
    extra: sweep(0.5, 30),
  },
];
