/**
 * Hash router: `#/` (home), `#/labs`, `#/lab/<id>`, `#/methods`, `#/method/<id>`, `#/research`,
 * `#/research/<study id>`, `#/python`, `#/dev` (component catalog).
 * Query parameters after `?` in the hash carry shareable view state (see useUrlState).
 */
import { useSyncExternalStore } from 'react';

export type Route =
  | { name: 'home' }
  | { name: 'labs' }
  | { name: 'lab'; id: string }
  | { name: 'methods' }
  | { name: 'method'; id: string }
  | { name: 'study'; id: string }
  | { name: 'research' }
  | { name: 'python' }
  | { name: 'dev' }
  | { name: 'notFound'; path: string };

const PAGES = ['labs', 'methods', 'research', 'python', 'dev'] as const;

const EVENT = 'numopt:hashstate';

function subscribe(cb: () => void) {
  window.addEventListener('hashchange', cb);
  window.addEventListener(EVENT, cb);
  return () => {
    window.removeEventListener('hashchange', cb);
    window.removeEventListener(EVENT, cb);
  };
}

const getHash = () => window.location.hash;

export function splitHash(hash: string): { path: string; query: string } {
  const h = hash.replace(/^#/, '');
  const i = h.indexOf('?');
  const path = (i < 0 ? h : h.slice(0, i)) || '/';
  return { path: path.startsWith('/') ? path : `/${path}`, query: i < 0 ? '' : h.slice(i + 1) };
}

export function parseRoute(hash: string): Route {
  const { path } = splitHash(hash);
  if (path === '/' || path === '') return { name: 'home' };
  const page = path.replace(/^\/|\/$/g, '');
  if ((PAGES as readonly string[]).includes(page)) return { name: page as (typeof PAGES)[number] };
  const m = /^\/lab\/([a-z0-9-]+)\/?$/.exec(path);
  if (m) return { name: 'lab', id: m[1] };
  const mm = /^\/method\/([a-z0-9_]+)\/?$/.exec(path);
  if (mm) return { name: 'method', id: mm[1] };
  const st = /^\/research\/([a-z0-9-]+)\/?$/.exec(path);
  if (st) return { name: 'study', id: st[1] };
  return { name: 'notFound', path };
}

export function useHash(): string {
  return useSyncExternalStore(subscribe, getHash, () => '');
}

export function useRoute(): Route {
  return parseRoute(useHash());
}

export function navigate(path: string): void {
  window.location.hash = path.startsWith('#') ? path : `#${path}`;
}

export const href = (path: string) => `#${path}`;

/** Replace the hash query without adding a history entry (and notify subscribers). */
export function replaceQuery(params: URLSearchParams): void {
  const { path } = splitHash(window.location.hash);
  const q = params.toString();
  const next = `#${path}${q ? `?${q}` : ''}`;
  if (next !== window.location.hash) {
    window.history.replaceState(window.history.state, '', next);
    window.dispatchEvent(new Event(EVENT));
  }
}

/** The hash query of the current location (read once, e.g. a Methods page filter). */
export function hashQuery(hash: string): URLSearchParams {
  return new URLSearchParams(splitHash(hash).query);
}
