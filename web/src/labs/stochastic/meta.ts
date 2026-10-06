import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'stochastic',
  title: 'Stochastic gradients',
  group: 'Optimization',
  problem: '\\min \\tfrac1n\\sum_i f_i(\\mathbf{w})',
  pitch: 'SGD, momentum, Adam and the variance-reduced SVRG, SAGA, SAG on noisy finite sums.',
  families: ['stochastic'],
  problemKinds: ['stochastic'],
  order: 5,
  icon: 'speed',
};

export default meta;
