import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'combinatorial',
  title: 'Combinatorial',
  group: 'Constrained & discrete',
  problem: '\\mathbf{x} \\in \\{0, 1\\}^n',
  pitch: 'Knapsack and TSP by dynamic programming, branch and bound, and heuristics.',
  families: ['combinatorial'],
  problemKinds: ['combinatorial'],
  order: 2,
  icon: 'grid',
};

export default meta;
