/**
 * Theme: 'system' follows `prefers-color-scheme`; 'light'/'dark' set `<html data-theme>`.
 * The preference is a per-viewer convenience kept in localStorage (guarded; may be unavailable).
 */
import { useMemo, useSyncExternalStore } from 'react';
import { readChartColors, type ChartColors, type ThemeMode } from './colors';

export type ThemePreference = 'system' | ThemeMode;
const KEY = 'numopt.theme';

const listeners = new Set<() => void>();
let preference: ThemePreference = readStored();

function readStored(): ThemePreference {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'light' || v === 'dark' ? v : 'system';
  } catch {
    return 'system';
  }
}

const media = () =>
  typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;

/** The OS color scheme, ignoring the viewer's explicit choice. */
export function systemTheme(): ThemeMode {
  return media()?.matches ? 'dark' : 'light';
}

function resolved(): ThemeMode {
  if (preference !== 'system') return preference;
  return media()?.matches ? 'dark' : 'light';
}

function apply() {
  const root = document.documentElement;
  if (preference === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', preference);
}

function emit() {
  for (const l of listeners) l();
}

export function initTheme(): void {
  apply();
  media()?.addEventListener('change', () => {
    if (preference === 'system') emit();
  });
}

let fadeTimer = 0;
/** Cross-fade surfaces for one theme switch (no-op under reduced motion: tokens are 0 ms). */
function fadeTheme() {
  const root = document.documentElement;
  root.setAttribute('data-theme-fading', '');
  window.clearTimeout(fadeTimer);
  fadeTimer = window.setTimeout(() => root.removeAttribute('data-theme-fading'), 260);
}

export function setThemePreference(p: ThemePreference): void {
  if (typeof document !== 'undefined') fadeTheme();
  preference = p;
  try {
    if (p === 'system') localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, p);
  } catch {
    /* storage unavailable: keep the in-memory preference */
  }
  apply();
  emit();
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

let snapshot = { preference, mode: 'light' as ThemeMode };
function getSnapshot() {
  const mode = resolved();
  if (snapshot.preference !== preference || snapshot.mode !== mode) snapshot = { preference, mode };
  return snapshot;
}

export function useTheme(): {
  preference: ThemePreference;
  mode: ThemeMode;
  setPreference: (p: ThemePreference) => void;
} {
  const s = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return { ...s, setPreference: setThemePreference };
}

/** Chart tokens for canvas drawing; recomputed when the resolved theme changes. */
export function useChartColors(): ChartColors {
  const { mode } = useTheme();
  // `mode` is the cache key: tokens.css changes values when the theme changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => readChartColors(), [mode]);
}
