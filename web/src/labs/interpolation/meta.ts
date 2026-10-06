import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'interpolation',
  title: 'Interpolation',
  group: 'Numerical analysis',
  problem: 'p(x_i) = y_i',
  pitch: 'Lagrange, Newton and splines through the same points — and Runge.',
  families: ['interpolation'],
  problemKinds: ['data'],
  order: 2,
  icon: 'layers',
};

export default meta;
