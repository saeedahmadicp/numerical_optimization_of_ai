/**
 * The lab registry — discovered, never edited by hand.
 *
 *   src/labs/<id>/meta.ts    default export: LabMeta (eager: tiny, needed by the home page)
 *   src/labs/<id>/index.tsx  default export: the lab component (lazy: its own chunk)
 *
 * A folder with a `meta.ts` and no `index.tsx` is listed as "planned". Method counts come from
 * `src/generated/registry.json` at build time (`CATALOG.byFamily`), so the home page never loads a
 * lab's code to count its methods. See web/README.md, "How to build a lab".
 */
import { lazy, type ComponentType, type LazyExoticComponent } from 'react';
import { CATALOG } from '../app/catalog';
import { pickLabForKind } from '../app/labPick';
import { LAB_GROUPS, type LabGroup, type LabMeta, type LabStatus } from './types';

export { LAB_GROUPS, type LabGroup, type LabMeta, type LabStatus } from './types';

export interface LabEntry extends Omit<LabMeta, 'status' | 'problemKinds' | 'order'> {
  status: LabStatus;
  /** Lazy lab component (absent for planned labs). */
  component?: LazyExoticComponent<ComponentType>;
  /** Starts loading the lab's chunk (hover/focus prefetch). */
  preload?: () => Promise<unknown>;
  /** Methods of the lab's families in the Python reference. */
  methodCount: number;
  problemKinds: string[];
  order: number;
}

const metas = import.meta.glob<LabMeta>('./*/meta.ts', { eager: true, import: 'default' });
const components = import.meta.glob<{ default: ComponentType }>('./*/index.tsx');

const folderOf = (path: string) => path.split('/')[1];

/** Build entries from discovered modules (exported for tests). */
export function buildLabs(
  metaModules: Record<string, LabMeta>,
  componentLoaders: Record<string, () => Promise<{ default: ComponentType }>>,
  byFamily: Record<string, number>,
): LabEntry[] {
  const loaders = new Map(Object.entries(componentLoaders).map(([p, l]) => [folderOf(p), l]));
  const entries = Object.entries(metaModules).map(([path, meta]): LabEntry => {
    const folder = folderOf(path);
    if (meta.id !== folder)
      console.warn(`[labs] ${path}: meta.id "${meta.id}" differs from its folder "${folder}"`);
    const load = loaders.get(folder);
    const status: LabStatus = load ? (meta.status ?? 'open') : 'planned';
    return {
      ...meta,
      id: folder,
      status,
      component: load ? lazy(load) : undefined,
      preload: load,
      methodCount: meta.families.reduce((n, f) => n + (byFamily[f] ?? 0), 0),
      problemKinds: meta.problemKinds ?? [],
      order: meta.order ?? 0,
    };
  });
  const g = (l: LabEntry) => LAB_GROUPS.indexOf(l.group);
  return entries.sort((a, b) => g(a) - g(b) || a.order - b.order || a.title.localeCompare(b.title));
}

export const LABS: LabEntry[] = buildLabs(metas, components, CATALOG.byFamily);

export function getLab(id: string): LabEntry | undefined {
  return LABS.find((l) => l.id === id);
}

export function labsInGroup(group: LabGroup): LabEntry[] {
  return LABS.filter((l) => l.group === group);
}

/** The lab that shows a Python family (first match in syllabus order). */
export function labForFamily(family: string): LabEntry | undefined {
  return LABS.find((l) => (l.families as string[]).includes(family));
}

/**
 * The lab a problem of this kind opens in: the open lab named after the kind (an `unconstrained`
 * problem opens the unconstrained lab, not the line-search lab that also uses it), else the
 * first open lab that uses the kind, else the first such lab.
 */
export function labForProblemKind(kind: string): LabEntry | undefined {
  return pickLabForKind(LABS, kind);
}

/** Methods the lab compares, from the Python registry (build time). Kept async for old callers. */
export async function availableMethods(lab: LabEntry): Promise<number> {
  return lab.methodCount;
}
