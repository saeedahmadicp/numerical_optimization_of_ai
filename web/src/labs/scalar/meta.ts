import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'scalar',
  title: '1-D minimization',
  group: 'Optimization',
  problem: '\\min_{a \\le x \\le b} f(x)',
  pitch: 'Golden section, Fibonacci and Brent shrink a bracket around a minimum.',
  families: ['scalar'],
  problemKinds: ['scalar_min'],
  order: 0,
  icon: 'target',
};

export default meta;
