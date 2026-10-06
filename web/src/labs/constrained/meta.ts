import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'constrained',
  title: 'Constrained',
  group: 'Constrained & discrete',
  problem: '\\min f(\\mathbf{x}) \\;\\text{s.t.}\\; c(\\mathbf{x}) \\le 0',
  pitch: 'Projected gradient, Frank–Wolfe, penalties, barriers, augmented Lagrangians and SQP.',
  families: ['constrained'],
  problemKinds: ['constrained'],
  order: 0,
  icon: 'target',
};

export default meta;
