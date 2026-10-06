import { useSyncExternalStore } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';
const mql = () =>
  typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(QUERY) : null;

function subscribe(cb: () => void) {
  const m = mql();
  m?.addEventListener('change', cb);
  return () => m?.removeEventListener('change', cb);
}

export const prefersReducedMotion = () => mql()?.matches ?? false;

/** True when the viewer asked the OS for reduced motion. Animations then jump instead of tween. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, prefersReducedMotion, () => false);
}
