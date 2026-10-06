/**
 * "Try this" presets of the systems lab. Every number in a note is a fact of the Python run (and
 * of the port, which replays it bit for bit); tests/systems/presets.test.ts checks each one.
 *
 * A preset that does not choose the focused method clears `fm`, so a focus left by an earlier
 * preset does not carry over.
 */
import type { LabPreset } from '../_shell';

export const PRESETS: LabPreset[] = [
  {
    id: 'leaves',
    title: 'Newton leaves the window',
    note:
      'From (−1, 1.25) Newton bounces for nine steps, mostly below the window. At 𝐱₉, ' +
      'x + y = 0.019, so det J = sin(x + y) ≈ 0.02, and the next step lands near the root ' +
      '(5π/6 − 6π, 5π/6 + 4π), far outside. Turn on damping: Newton then reaches (π/6, π/6) in 7 steps.',
    problem: 'trig_system',
    methods: [{ id: 'newton_system', slot: 0, params: {} }],
    start: [-1, 1.25],
    extra: { f: 'norm', fm: '' },
  },
  {
    id: 'trap',
    title: 'A minimum of ‖F‖ that is not a root',
    note:
      'Damped Newton slides into the valley y ≈ −0.897, where det J = 6y² − 8y − 12 vanishes, and ' +
      'its line search fails at (13.55, −0.897) with ½‖F‖² = 29.1. The non-root minimum of ½‖F‖² ' +
      'is at (11.41, −0.897). Without damping, Newton reaches the root (5, 4) in 43 steps.',
    problem: 'freudenstein_roth',
    methods: [{ id: 'newton_system', slot: 0, params: { damping: true } }],
    extra: { f: 'norm', fm: '' },
  },
  {
    id: 'drift',
    title: "Broyden's model drifts",
    note:
      'On the Rosenbrock gradient Newton needs 7 steps. The zero lines of Broyden’s model (orange) ' +
      'drift away from those of the exact J (gray). Broyden creeps along the valley and spends its ' +
      '100-step budget, stopping at (−0.31, 0.09) with ‖F‖ = 3.5.',
    problem: 'rosenbrock_system',
    methods: [
      { id: 'newton_system', slot: 0, params: {} },
      { id: 'broyden', slot: 1, params: {} },
    ],
    extra: { f: 'norm', fm: 'broyden' },
  },
  {
    id: 'basins',
    title: 'Which root does Newton find?',
    note: 'Every start point colored by the root Newton reaches. The boundaries fold where det J = sin(x + y) vanishes.',
    problem: 'trig_system',
    methods: [{ id: 'newton_system', slot: 0, params: {} }],
    extra: { f: 'basins', fm: '' },
  },
];
