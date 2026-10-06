import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'linalg',
  title: 'Linear systems',
  group: 'Equations',
  problem: 'A\\mathbf{x} = \\mathbf{b}',
  pitch: 'Elimination, factorizations and iterations for Ax = b, pivot by pivot, sweep by sweep.',
  families: ['linalg'],
  problemKinds: ['linalg'],
  order: 2,
  icon: 'grid',
};

export default meta;
