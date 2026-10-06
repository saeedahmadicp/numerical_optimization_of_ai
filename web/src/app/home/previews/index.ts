/**
 * Lazy loader for the lab previews: one chunk per lab (`./labs/<lab id>.ts`, with only the method
 * ports and problems that preview runs). A preview is built once per page load and shared by the
 * hero and the lab card.
 */
import type { Preview, PreviewModule } from './types';

const LOADERS = import.meta.glob<PreviewModule>('./labs/*.ts');
const byId = new Map(
  Object.entries(LOADERS).map(([path, load]) => [path.replace(/^.*\/(.+)\.ts$/, '$1'), load]),
);
const CACHE = new Map<string, Promise<Preview | null>>();

export function hasPreview(lab: string): boolean {
  return byId.has(lab);
}

/**
 * Starts the download of a preview's chunk (and the chunks it imports) without building the
 * preview: the module is evaluated, but its runs wait for `loadPreview`. The hero uses this to
 * fetch its later scenes in parallel and build them one at a time when the main thread is idle.
 */
export function prefetchPreview(lab: string): void {
  if (CACHE.has(lab)) return;
  byId
    .get(lab)?.()
    .catch(() => {
      // loadPreview reports the failure when the preview is built.
    });
}

export function loadPreview(lab: string): Promise<Preview | null> {
  let p = CACHE.get(lab);
  if (!p) {
    const load = byId.get(lab);
    p = load
      ? load()
          .then((m) => m.default())
          .catch((e: unknown) => {
            console.warn(`[home] preview ${lab} failed`, e);
            return null;
          })
      : Promise.resolve(null);
    CACHE.set(lab, p);
  }
  return p;
}
