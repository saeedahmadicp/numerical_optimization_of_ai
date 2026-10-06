/**
 * Defaults and "Try this" presets of the 1-D minimization lab. Every preset names the phenomenon
 * it reveals; the numbers in the notes are the Python reference's (numopt.scalar.methods).
 */
import type { LabPreset, MethodSelection } from '../_shell';

/** First view: golden section's fixed ratio against Brent's parabolic steps on a real model. */
export const DEFAULT_PROBLEM = 'drug_concentration';
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'golden_section', slot: 0, params: {} },
  { id: 'brent_minimize', slot: 1, params: {} },
];

/** `br=a,b` (the bracket) is set by every preset, so a stale bracket never leaks into one. */
export const PRESETS: LabPreset[] = [
  {
    id: 'thirds',
    title: 'Thirds lose to the golden ratio',
    note: 'Ternary search re-evaluates both probes; golden section reuses one. Per evaluation: 0.816 against 0.618.',
    problem: 'quadratic_1d',
    methods: [
      { id: 'golden_section', slot: 0, params: {} },
      { id: 'ternary_search', slot: 1, params: {} },
    ],
    extra: { br: '0,5', y: 'width' },
  },
  {
    id: 'fixed-end',
    title: 'A fixed end slows the parabolas',
    note: 'xᵣ = 2 stays in the triple for 28 steps while the vertex creeps in from the left (31 steps); Brent fits through its best points (10).',
    problem: 'quartic_1d',
    methods: [
      { id: 'parabolic_interpolation', slot: 0, params: {} },
      { id: 'brent_minimize', slot: 1, params: {} },
    ],
    extra: { br: '0.5,2', y: 'error' },
  },
  {
    id: 'kink',
    title: 'A kink defeats Newton',
    note: 'At x⋆ = 0.3, f′ jumps by 2: Newton bounces between −1 and 1 for 100 steps; elimination does not care.',
    problem: 'abs_shifted',
    methods: [
      { id: 'newton_1d', slot: 0, params: {} },
      { id: 'golden_section', slot: 1, params: {} },
    ],
    start: -0.5,
    extra: { br: '-1,1', y: 'error' },
  },
  {
    id: 'rounding',
    title: 'Rounding stops dichotomous search',
    note: 'Probes 10⁻¹³ apart: f(x₁) and f(x₂) agree to the last bits while the bracket is still 0.094 wide.',
    problem: 'drug_concentration',
    methods: [
      { id: 'dichotomous_search', slot: 0, params: { xtol: 1e-10, delta_ratio: 0.001 } },
      { id: 'golden_section', slot: 1, params: {} },
    ],
    extra: { br: '0,12', y: 'width' },
  },
];
