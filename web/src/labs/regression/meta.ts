import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'regression',
  title: 'Regression',
  group: 'Data',
  problem:
    '\\min_{\\boldsymbol\\beta} \\sum_i \\rho(y_i - \\mathbf{a}_i^{\\top}\\boldsymbol\\beta)',
  pitch:
    'Least squares, ridge, Huber, LAD, Theil–Sen and minimax fits through points you can drag.',
  families: ['regression'],
  problemKinds: ['data'],
  order: 0,
  icon: 'grid',
};

export default meta;
