/**
 * Autoplay that waits for the landscape, driven by Contour2D's `onFieldReady`: the player is
 * created with `autoplay: false`, and each new set of traces starts playing once the contour
 * field of `fieldKey` is on screen, so the first iterates never move over a blank plot and the
 * field never pops in under a moving run. A field already shown (a parameter change on the same
 * problem) starts the run at once; a field that never reports starts it after FIELD_WAIT_MS.
 * `fieldKey` null means there is no 2-D field to wait for (the 3-D view): start at once. Under
 * reduced motion the player shows the final step instead.
 *
 *   const player = useTracePlayer(traces, { autoplay: false });
 *   const onFieldReady = useStartOnField(player, traces, problem.id);
 *   <Contour2D cacheKey={problem.id} onFieldReady={onFieldReady} … />
 *
 * (src/play/useStartWhenDrawn.ts is the polling variant for a plot that does not report.)
 */
import { useCallback, useEffect, useLayoutEffect, useRef } from 'react';
import { FIELD_WAIT_MS } from '../../play/useStartWhenDrawn';
import type { TracePlayer } from '../../play/useTracePlayer';

export function useStartOnField(
  player: TracePlayer,
  traces: unknown,
  fieldKey: string | null,
): () => void {
  const playerRef = useRef(player);
  const keyRef = useRef(fieldKey);
  useLayoutEffect(() => {
    playerRef.current = player;
    keyRef.current = fieldKey;
  });
  /** The field key whose raster has been shown. */
  const shown = useRef<string | null>(null);
  /** The current traces have not been started yet. */
  const armed = useRef(false);

  const start = useCallback(() => {
    if (!armed.current) return;
    armed.current = false;
    const p = playerRef.current;
    // The viewer may have pressed Play or scrubbed meanwhile: then leave the player alone.
    if (!p.playing && p.t === 0) p.play();
  }, []);

  useEffect(() => {
    const p = playerRef.current;
    if (p.reducedMotion) {
      armed.current = false;
      p.toEnd();
      return;
    }
    armed.current = true;
    if (fieldKey === null || shown.current === fieldKey) {
      start();
      return;
    }
    const timer = window.setTimeout(start, FIELD_WAIT_MS);
    return () => window.clearTimeout(timer);
  }, [traces, fieldKey, start]);

  // Contour2D reports in its own effects, which run before this component's effects: a cached
  // field of a new problem is recorded here and started by the effect above.
  return useCallback(() => {
    shown.current = keyRef.current;
    start();
  }, [start]);
}
