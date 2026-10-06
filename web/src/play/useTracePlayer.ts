import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { advance, baseRate, localT, maxStep, SPEEDS } from './timeline';
import { usePrefersReducedMotion } from './reducedMotion';

export interface TracePlayerOptions {
  /**
   * Start playing whenever the traces change (default true). Ignored under
   * prefers-reduced-motion: the player then stays paused on the final step until the viewer
   * presses Play, and steps at a capped, readable rate.
   */
  autoplay?: boolean;
  /** Initial speed index into SPEEDS (default 2 → 1×). */
  speedIndex?: number;
  /** Restart from 0 when the traces change (default true). */
  resetOnChange?: boolean;
  /** Loop back to 0 after a pause at the end (hero animations). Off under reduced motion. */
  loop?: boolean;
}

/** Steps per second at 1× under prefers-reduced-motion (each step stays readable). */
export const REDUCED_MOTION_RATE = 6;

export interface TracePlayer {
  /** Integer step under the playhead (= floor(t)). */
  k: number;
  /** Continuous playhead in [0, maxK]. */
  t: number;
  maxK: number;
  lengths: number[];
  playing: boolean;
  /** Speed multiplier (one of SPEEDS). */
  speed: number;
  atEnd: boolean;
  reducedMotion: boolean;
  play(): void;
  pause(): void;
  toggle(): void;
  /** Move ±n whole steps (pauses). */
  step(delta: number): void;
  seek(t: number): void;
  reset(): void;
  toEnd(): void;
  setSpeed(multiplier: number): void;
  faster(): void;
  slower(): void;
  /** Local time of method i (stops at its own last step). */
  localT(i: number): number;
  /** Local integer step of method i. */
  localK(i: number): number;
}

type Lengthy = { length: number } | number;

/**
 * Time-based playback synchronized across compared methods.
 *
 * `traces` may be arrays (their `.length` is used) or plain lengths. Progression uses
 * requestAnimationFrame and wall-clock time, so speed is independent of frame rate. With
 * prefers-reduced-motion the playhead jumps whole steps instead of interpolating.
 */
export function useTracePlayer(
  traces: readonly Lengthy[],
  options: TracePlayerOptions = {},
): TracePlayer {
  const { speedIndex = 2, resetOnChange = true } = options;
  const lengths = useMemo(
    () => traces.map((tr) => (typeof tr === 'number' ? tr : tr.length)),
    [traces],
  );
  const maxK = maxStep(lengths);
  const reducedMotion = usePrefersReducedMotion();
  // Reduced motion: no autoplay and no loop; show the outcome (the final step) instead.
  const autoplay = (options.autoplay ?? true) && !reducedMotion;
  const loop = (options.loop ?? false) && !reducedMotion;
  const showEnd = (options.autoplay ?? true) && reducedMotion;

  const [t, setT] = useState(() => (showEnd ? maxK : 0));
  const [playing, setPlaying] = useState(autoplay);
  const [speed, setSpeedState] = useState<number>(SPEEDS[speedIndex] ?? 1);

  // New traces → restart (keeps a lab's "change a param, watch it re-run" loop tight).
  // Adjusting state during render (instead of in an effect) avoids a stale frame.
  const [prevTraces, setPrevTraces] = useState(traces);
  if (prevTraces !== traces) {
    setPrevTraces(traces);
    if (resetOnChange) {
      setT(showEnd ? maxK : 0);
      setPlaying(autoplay);
    } else if (t > maxK) {
      setT(maxK);
    }
  }

  const tRef = useRef(0);
  const accRef = useRef(0);
  const raf = useRef(0);
  const last = useRef<number | null>(null);
  const holdUntil = useRef(0);
  const pendingRestart = useRef(false);

  useLayoutEffect(() => {
    tRef.current = t;
  }, [t]);
  useLayoutEffect(() => {
    accRef.current = 0;
  }, [traces]);

  const setBoth = useCallback((v: number) => {
    tRef.current = v;
    setT(v);
  }, []);

  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    const rate =
      (reducedMotion ? Math.min(baseRate(maxK), REDUCED_MOTION_RATE) : baseRate(maxK)) * speed;
    const tick = (now: number) => {
      if (last.current === null) last.current = now;
      const dt = Math.min(0.1, (now - last.current) / 1000);
      last.current = now;
      if (now < holdUntil.current) {
        raf.current = requestAnimationFrame(tick);
        return;
      }
      if (pendingRestart.current) {
        // The hold on the final frame is over: start again from k = 0.
        pendingRestart.current = false;
        accRef.current = 0;
        setBoth(0);
        holdUntil.current = now + 450;
        raf.current = requestAnimationFrame(tick);
        return;
      }
      if (tRef.current >= maxK) {
        if (loop) {
          pendingRestart.current = true;
          holdUntil.current = now + 2400;
          raf.current = requestAnimationFrame(tick);
          return;
        }
        setPlaying(false);
        return;
      }
      const next = advance(tRef.current, dt, rate, maxK, reducedMotion, accRef.current);
      accRef.current = next.acc;
      if (next.t !== tRef.current) setBoth(next.t);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, speed, maxK, reducedMotion, loop, setBoth]);

  const play = useCallback(() => {
    if (tRef.current >= maxK) setBoth(0);
    holdUntil.current = 0;
    pendingRestart.current = false;
    setPlaying(true);
  }, [maxK, setBoth]);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => (playing ? pause() : play()), [playing, pause, play]);
  const seek = useCallback((v: number) => setBoth(Math.max(0, Math.min(maxK, v))), [maxK, setBoth]);
  const step = useCallback(
    (d: number) => {
      setPlaying(false);
      const base = d > 0 ? Math.floor(tRef.current + 1e-9) : Math.ceil(tRef.current - 1e-9);
      seek(base + d);
    },
    [seek],
  );
  const reset = useCallback(() => {
    setPlaying(false);
    accRef.current = 0;
    setBoth(0);
  }, [setBoth]);
  const toEnd = useCallback(() => {
    setPlaying(false);
    setBoth(maxK);
  }, [maxK, setBoth]);
  const setSpeed = useCallback((m: number) => setSpeedState(m), []);
  const faster = useCallback(() => setSpeedState((s) => SPEEDS.find((x) => x > s) ?? s), []);
  const slower = useCallback(
    () => setSpeedState((s) => [...SPEEDS].reverse().find((x) => x < s) ?? s),
    [],
  );

  const tc = Math.min(t, maxK);
  return {
    k: Math.floor(tc + 1e-9),
    t: tc,
    maxK,
    lengths,
    playing,
    speed,
    atEnd: tc >= maxK,
    reducedMotion,
    play,
    pause,
    toggle,
    step,
    seek,
    reset,
    toEnd,
    setSpeed,
    faster,
    slower,
    localT: (i: number) => localT(tc, lengths[i] ?? 1),
    localK: (i: number) => Math.floor(localT(tc, lengths[i] ?? 1) + 1e-9),
  };
}

const TEXT_ENTRY = 'input, textarea, select, [contenteditable="true"]';
/** Widgets that use the arrow keys (and Home/End) themselves. */
const ARROW_OWNERS = [
  '[role="radio"]',
  '[role="radiogroup"]',
  '[role="slider"]',
  '[role="tab"]',
  '[role="tablist"]',
  '[role="listbox"]',
  '[role="option"]',
  '[role="combobox"]',
  '[role="menu"]',
  '[role="menuitem"]',
  '[role="grid"]',
  '[role="tree"]',
  '[aria-haspopup]:not([aria-haspopup="false"])',
  '[data-own-keys]',
].join(', ');
const OWN_SPACE = `button, [role="button"], [role="switch"], summary, a, ${ARROW_OWNERS}`;
/** Region where Home/End drive the player (LabShell marks the stage and its playback bar). */
export const PLAYER_SCOPE_ATTR = 'data-player-scope';

/**
 * Playback shortcuts: Space play/pause, ←/→ step (Shift = 10), Home/End, R reset, +/− speed.
 *
 * - Ignored while typing and on widgets that own the arrow keys (radios, sliders, tabs,
 *   listboxes, popup triggers, anything with `data-own-keys`).
 * - Space is left to focused buttons and links.
 * - Home/End only act inside an element marked `data-player-scope` (the stage), so they keep
 *   scrolling the page everywhere else.
 */
export function usePlayerKeyboard(player: TracePlayer, enabled = true): void {
  const ref = useRef(player);
  useEffect(() => {
    ref.current = player;
  });
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as Element | null;
      if (target?.closest?.(TEXT_ENTRY)) return;
      const navKey = /^(Arrow(Left|Right|Up|Down)|Home|End|PageUp|PageDown)$/.test(e.key);
      if (navKey && target?.closest?.(ARROW_OWNERS)) return;
      if ((e.key === 'Home' || e.key === 'End') && !target?.closest?.(`[${PLAYER_SCOPE_ATTR}]`))
        return;
      const p = ref.current;
      switch (e.key) {
        case ' ':
          if (target?.closest?.(OWN_SPACE)) return;
          p.toggle();
          break;
        case 'ArrowRight':
          p.step(e.shiftKey ? 10 : 1);
          break;
        case 'ArrowLeft':
          p.step(e.shiftKey ? -10 : -1);
          break;
        case 'Home':
          p.reset();
          break;
        case 'End':
          p.toEnd();
          break;
        case 'r':
        case 'R':
          p.reset();
          break;
        case '+':
        case '=':
          p.faster();
          break;
        case '-':
        case '_':
          p.slower();
          break;
        default:
          return;
      }
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [enabled]);
}
