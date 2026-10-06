/** Interpolation lab: defaults, "Try this" presets, URL codecs and labels. */
import { codecs } from '../../app/useUrlState';
import { URL_KEYS, type LabPreset, type MethodSelection } from '../_shell';
import type { Layout } from './geometry';
import type { StripMode } from './StripPlot';

/**
 * First view: Runge's function on its 11 equispaced nodes. Newton's form adds one node per
 * step (the divided-difference table grows beside it) and swings to an error of 1.9 near ±1,
 * while the not-a-knot spline through the same nodes stays within 0.022.
 */
export const DEFAULT_PROBLEM = 'runge_equispaced';
export const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'newton_divided_differences', slot: 0, params: {} },
  { id: 'cubic_spline_not_a_knot', slot: 1, params: {} },
];

/** URL keys. The edited nodes use the shell's start-point key, so presets and a problem change clear them. */
export const KEYS = {
  layout: 'nl',
  count: 'n',
  nodes: URL_KEYS.start,
  snap: 'snap',
  strip: 'v',
} as const;

export const LAYOUT_CODEC = codecs.oneOf<Layout>(['data', 'equi', 'cheb']);
export const STRIP_CODEC = codecs.oneOf<StripMode>(['error', 'omega', 'basis', 'poles']);

export const PRESETS: LabPreset[] = [
  {
    id: 'runge',
    title: "Runge's phenomenon on 15 equispaced nodes",
    note: 'The degree-14 interpolant misses f by 7.2 near ±1; the not-a-knot spline through the same nodes by 0.0025.',
    problem: 'runge_equispaced',
    methods: [
      { id: 'newton_divided_differences', slot: 0, params: {} },
      { id: 'cubic_spline_not_a_knot', slot: 1, params: {} },
    ],
    extra: { [KEYS.layout]: 'equi', [KEYS.count]: '15', [KEYS.strip]: 'error' },
  },
  {
    id: 'chebyshev',
    title: 'Chebyshev nodes tame it',
    note: 'Same f, same degree, nodes clustered at the ends: max|ω| falls from 1.9×10⁻³ to 6.1×10⁻⁵ and the error from 7.2 to 0.047.',
    problem: 'runge_equispaced',
    methods: [
      { id: 'barycentric', slot: 0, params: {} },
      { id: 'cubic_spline_not_a_knot', slot: 1, params: {} },
    ],
    extra: { [KEYS.layout]: 'cheb', [KEYS.count]: '15', [KEYS.strip]: 'omega' },
  },
  {
    id: 'basis',
    title: 'The basis behind the polynomial',
    note: 'Each ℓⱼ is 1 at its own node and 0 at the other eight; p = Σ yⱼℓⱼ gains one term per step.',
    problem: 'sine_samples',
    methods: [{ id: 'lagrange', slot: 0, params: {} }],
    extra: { [KEYS.layout]: 'data', [KEYS.count]: '9', [KEYS.strip]: 'basis' },
  },
  {
    id: 'jump',
    title: 'A jump: splines ring, PCHIP does not',
    note: 'The natural spline swings to −0.107 and 1.107 around the step; PCHIP and the linear spline stay inside [0, 1].',
    problem: 'step_data',
    methods: [
      { id: 'cubic_spline_natural', slot: 0, params: {} },
      { id: 'pchip', slot: 1, params: {} },
      { id: 'linear_spline', slot: 2, params: {} },
    ],
    extra: { [KEYS.layout]: 'data', [KEYS.count]: '12', [KEYS.strip]: 'error' },
  },
  {
    id: 'aaa',
    title: "AAA finds the poles of Runge's function",
    note: 'Three support points recover 1/(1 + 25x²) to 6.7×10⁻¹⁶, with poles at ±0.2i; the degree-14 polynomial misses f by 7.2.',
    problem: 'runge_equispaced',
    methods: [
      { id: 'newton_divided_differences', slot: 0, params: {} },
      { id: 'aaa', slot: 1, params: {} },
    ],
    extra: { [KEYS.layout]: 'equi', [KEYS.count]: '15', [KEYS.strip]: 'poles' },
  },
  {
    id: 'floater-hormann',
    title: 'Floater–Hormann: no Runge, no poles',
    note: "On 21 equispaced nodes the polynomial misses f by 58; the d = 3 blend by 0.0028, close to the spline's 0.0032.",
    problem: 'runge_equispaced',
    methods: [
      { id: 'barycentric', slot: 0, params: {} },
      { id: 'floater_hormann', slot: 1, params: { d: 3 } },
      { id: 'cubic_spline_not_a_knot', slot: 2, params: {} },
    ],
    extra: { [KEYS.layout]: 'equi', [KEYS.count]: '21', [KEYS.strip]: 'error' },
  },
  {
    id: 'aaa-noise',
    title: 'AAA on noisy data: poles between the samples',
    note: 'AAA interpolates 11 of the 20 noisy samples and puts 7 certified real poles in [0, 10]; Floater–Hormann goes through all 20 with none.',
    problem: 'noisy_linear',
    methods: [
      { id: 'aaa', slot: 0, params: {} },
      { id: 'floater_hormann', slot: 1, params: { d: 3 } },
    ],
    extra: { [KEYS.layout]: 'data', [KEYS.count]: '20', [KEYS.strip]: 'poles' },
  },
];

/** Title of the Table tab for a method. */
export function tableTitle(methodId: string): string {
  switch (methodId) {
    case 'newton_divided_differences':
      return 'Divided differences';
    case 'neville':
      return 'Tableau';
    case 'chebyshev_interpolation':
      return 'Coefficients';
    case 'pchip':
      return 'Slopes';
    case 'linear_spline':
      return 'Pieces';
    case 'barycentric':
      return 'Weights';
    case 'lagrange':
      return 'Basis';
    case 'aaa':
      return 'Support points';
    case 'floater_hormann':
      return 'Weights';
    default:
      return 'System';
  }
}

/** What a spline step did, in words (the Steps table). */
export const STAGE_LABEL: Record<string, string> = {
  assemble: 'assemble T s = r',
  forward_sweep: 'forward sweep',
  back_substitution: 'back substitution',
  secants: 'secants mᵢ',
  slopes: 'slopes sᵢ',
  coefficients: 'pieces Sᵢ',
};

/** Node counts of the error-vs-n study. */
export const SWEEP_N = Array.from({ length: 38 }, (_, i) => i + 3);
