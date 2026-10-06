/**
 * Lab metadata. Every lab folder `src/labs/<id>/` has a `meta.ts` whose default export is a
 * `LabMeta`; `src/labs/index.ts` discovers them with `import.meta.glob`, so adding a lab never
 * touches a shared file. A folder without `index.tsx` is listed as "planned".
 */
import type { Family } from '../core/types';
import type { IconName } from '../ui/components/Icon';

export type LabGroup =
  'Equations' | 'Optimization' | 'Constrained & discrete' | 'Numerical analysis' | 'Data';

/** Home-page order of the syllabus groups. */
export const LAB_GROUPS: readonly LabGroup[] = [
  'Equations',
  'Optimization',
  'Constrained & discrete',
  'Numerical analysis',
  'Data',
];

/**
 * - `open`: built and announced (green "open" pill on the home page).
 * - `preview`: built and reachable by URL, listed like a planned lab until it is ready.
 * - `planned`: no `index.tsx` yet (the route shows the planned-lab page with its methods).
 */
export type LabStatus = 'open' | 'preview' | 'planned';

export interface LabMeta {
  /** URL id = folder name (`#/lab/<id>`), lowercase with hyphens. */
  id: string;
  /** Short title in sentence case ("Root finding"). */
  title: string;
  group: LabGroup;
  /** The problem the lab solves, in LaTeX (`f(x) = 0`, `\min_{\mathbf{x}} f(\mathbf{x})`). */
  problem: string;
  /** One line, ≤ 90 characters. Name only methods that ship. */
  pitch: string;
  /** Python families whose methods the lab compares (method counts come from the registry). */
  families: Family[];
  /** Python problem kinds the lab draws from (search sends a problem to the first such lab). */
  problemKinds?: string[];
  /** Default `open` once `index.tsx` exists; set `preview` to keep a built lab unannounced. */
  status?: Exclude<LabStatus, 'planned'>;
  /** Position inside its group (ascending; default 0, then title). */
  order?: number;
  icon?: IconName;
}
