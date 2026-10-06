import type { LabMeta } from '../types';

/** Lab metadata (discovered by src/labs/index.ts). The lab is "planned" until ./index.tsx exists. */
const meta: LabMeta = {
  id: 'least-squares',
  title: 'Nonlinear least squares',
  group: 'Optimization',
  problem: '\\min \\tfrac12 \\|\\mathbf{r}(\\mathbf{x})\\|_2^2',
  pitch: 'Gauss–Newton and Levenberg–Marquardt fit models to residuals.',
  families: ['least_squares'],
  problemKinds: ['least_squares'],
  order: 3,
  icon: 'layers',
};

export default meta;
