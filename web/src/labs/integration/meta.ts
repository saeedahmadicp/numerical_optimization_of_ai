import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'integration',
  title: 'Quadrature',
  group: 'Numerical analysis',
  problem: '\\int_a^b f(x)\\,dx',
  pitch: 'Trapezoid, Simpson, Romberg and Gauss rules fill the area under a curve.',
  families: ['integration'],
  problemKinds: ['calculus'],
  order: 0,
  icon: 'layers',
};

export default meta;
