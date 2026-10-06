import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'differentiation',
  title: 'Differentiation',
  group: 'Numerical analysis',
  problem: "f'(x_0),\\ f''(x_0)",
  pitch: 'Difference quotients, Richardson and the complex step: where truncation meets round-off.',
  families: ['differentiation'],
  problemKinds: ['calculus'],
  order: 1,
  icon: 'speed',
};

export default meta;
