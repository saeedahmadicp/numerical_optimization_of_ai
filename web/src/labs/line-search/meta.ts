import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'line-search',
  title: 'Line search',
  group: 'Optimization',
  problem: '\\min_{\\alpha > 0} \\varphi(\\alpha)',
  pitch: 'Armijo, Wolfe and Goldstein conditions decide how far to step.',
  families: ['line_search'],
  problemKinds: ['unconstrained'],
  order: 1,
  icon: 'arrowRight',
};

export default meta;
