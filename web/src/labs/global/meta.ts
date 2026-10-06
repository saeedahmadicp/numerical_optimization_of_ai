import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'global',
  title: 'Global',
  group: 'Optimization',
  problem: '\\min_{\\mathbf{x} \\in \\Omega} f(\\mathbf{x})',
  pitch: 'Annealing, swarms, DE, CMA-ES and basin hopping search multimodal landscapes.',
  families: ['global'],
  problemKinds: ['unconstrained'],
  order: 4,
  icon: 'target',
};

export default meta;
