import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'roots',
  title: 'Root finding',
  group: 'Equations',
  problem: 'f(x) = 0',
  pitch: 'Bisection to ITP, Newton to Müller: sixteen root finders race to a zero of f(x).',
  families: ['roots'],
  problemKinds: ['roots'],
  order: 0,
  icon: 'crosshair',
};

export default meta;
