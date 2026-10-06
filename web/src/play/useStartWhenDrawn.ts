/**
 * Autoplay that waits for the landscape. Create the player with `autoplay: false` and pass the
 * element that holds the contour canvas; each time the traces change, this hook polls the canvas
 * (one pixel row per frame) and starts playback once the field raster is on screen, so the first
 * iterates never animate over a blank or hatch-only plot and the field never pops in under a
 * moving run. It starts anyway after FIELD_WAIT_MS. Under reduced motion it shows the final step.
 *
 *   const player = useTracePlayer(traces, { autoplay: false });
 *   const host = useRef<HTMLDivElement>(null);
 *   useStartWhenDrawn(player, host, traces);
 *   <div ref={host}><Contour2D … /></div>
 *
 * A lab whose Contour2D reports `onFieldReady` can call `player.play()` from it instead; this hook
 * is the shared fallback that needs no callback.
 */
import { useEffect, useLayoutEffect, useRef, type RefObject } from 'react';
import type { TracePlayer } from './useTracePlayer';

/** Start anyway after this long (a field that never paints must not block playback). */
export const FIELD_WAIT_MS = 4000;

/** True when a canvas row holds more than one color (the field, not the empty surface). */
export function rowVaries(data: Uint8ClampedArray): boolean {
  for (let o = 4; o < data.length; o += 4)
    if (
      Math.abs(data[o] - data[0]) > 6 ||
      Math.abs(data[o + 1] - data[1]) > 6 ||
      Math.abs(data[o + 2] - data[2]) > 6
    )
      return true;
  return false;
}

function fieldDrawn(host: HTMLElement | null): boolean {
  const canvas = host?.querySelector('canvas');
  if (!canvas || canvas.width < 2 || canvas.height < 2) return false;
  try {
    const ctx = canvas.getContext('2d');
    if (!ctx) return true;
    const y = Math.floor(canvas.height / 2);
    return rowVaries(ctx.getImageData(0, y, canvas.width, 1).data);
  } catch {
    return true;
  }
}

export function useStartWhenDrawn(
  player: TracePlayer,
  host: RefObject<HTMLElement | null>,
  traces: unknown,
) {
  const ref = useRef(player);
  useLayoutEffect(() => {
    ref.current = player;
  });
  useEffect(() => {
    const p = ref.current;
    if (p.reducedMotion) {
      p.toEnd();
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    let frames = 0;
    const check = () => {
      frames += 1;
      // Skip the first frame: the canvas may still hold the previous problem's field.
      const ready = frames > 1 && fieldDrawn(host.current);
      if (ready || performance.now() - t0 > FIELD_WAIT_MS) {
        const q = ref.current;
        // The viewer may have pressed Play or scrubbed meanwhile: then leave the player alone.
        if (!q.playing && q.t === 0) q.play();
        return;
      }
      raf = requestAnimationFrame(check);
    };
    raf = requestAnimationFrame(check);
    return () => cancelAnimationFrame(raf);
  }, [traces, host]);
}
