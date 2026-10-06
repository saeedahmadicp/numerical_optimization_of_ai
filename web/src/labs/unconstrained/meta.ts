import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'unconstrained',
  title: 'Unconstrained',
  group: 'Optimization',
  problem: '\\min_{\\mathbf{x}} f(\\mathbf{x})',
  pitch: 'Gradient, Newton, quasi-Newton, CG, trust-region and simplex steps, drawn as taken.',
  families: ['unconstrained'],
  problemKinds: ['unconstrained'],
  order: 2,
  icon: 'layers',
};

export default meta;
