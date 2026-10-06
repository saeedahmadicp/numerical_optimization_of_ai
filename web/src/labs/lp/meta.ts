import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'lp',
  title: 'Linear & integer programming',
  group: 'Constrained & discrete',
  problem: '\\min \\mathbf{c}^{\\top}\\mathbf{x} \\;\\text{s.t.}\\; A\\mathbf{x} \\le \\mathbf{b}',
  pitch: 'Simplex walks the vertices; interior points cut through the middle.',
  families: ['lp'],
  problemKinds: ['lp'],
  order: 1,
  icon: 'cube',
};

export default meta;
