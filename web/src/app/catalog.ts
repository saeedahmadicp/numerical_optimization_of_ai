/**
 * The catalog of the Python reference, read from `src/generated/*.json` at build time.
 *
 * `CATALOG` (counts) is a few hundred bytes and safe to import anywhere. `loadCatalogIndex()`
 * lazily loads the method, problem and research lists (search, Methods, Research, planned labs).
 */
import counts from 'virtual:numopt/catalog';
import type { CatalogIndex } from './catalogTypes';

export type * from './catalogTypes';

export const CATALOG = counts;

let index: Promise<CatalogIndex> | null = null;

export function loadCatalogIndex(): Promise<CatalogIndex> {
  index ??= import('virtual:numopt/index').then((m) => ({
    methods: m.methods,
    problems: m.problems,
    research: m.research,
  }));
  return index;
}

/** The repository (links to the Python package, docs, research notes). */
export const REPO = 'https://github.com/saeedahmadicp/numerical_optimization_of_ai';
