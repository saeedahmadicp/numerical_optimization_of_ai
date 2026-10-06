import { use, useEffect, useState } from 'react';
import { loadCatalogIndex } from './catalog';
import type { CatalogIndex } from './catalogTypes';

/** The method / problem / research lists (lazy chunk); null while loading. */
export function useCatalogIndex(): CatalogIndex | null {
  const [index, setIndex] = useState<CatalogIndex | null>(null);
  useEffect(() => {
    let alive = true;
    loadCatalogIndex().then((ix) => alive && setIndex(ix));
    return () => {
      alive = false;
    };
  }, []);
  return index;
}

/**
 * The same lists for a page under a <Suspense> boundary: suspends (the route's PageLoading
 * fallback shows) until the module-cached index has loaded, so the page renders once, complete,
 * and nothing moves when the data arrives.
 */
export function useCatalogIndexNow(): CatalogIndex {
  return use(loadCatalogIndex());
}
