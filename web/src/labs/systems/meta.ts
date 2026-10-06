import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'systems',
  title: 'Nonlinear systems',
  group: 'Equations',
  problem: 'F(\\mathbf{x}) = \\mathbf{0}',
  pitch: 'Newton and Broyden solve F(x) = 0 where two curves cross.',
  families: ['systems'],
  problemKinds: ['systems'],
  order: 1,
  icon: 'crosshair',
};

export default meta;
