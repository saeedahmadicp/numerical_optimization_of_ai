/**
 * A lab in the gallery: the lab's signature geometry as a still (the end of a real run), then
 * title, pitch and method count. The whole card is the link. Hover or keyboard focus plays the
 * run once (the preview's `draw` over u = 0 → 1); reduced motion keeps the still.
 */
import { useEffect, useRef, useState } from 'react';
import type { LabEntry } from '../../labs';
import { int } from '../../core/format';
import { usePrefersReducedMotion } from '../../play/reducedMotion';
import { useChartColors } from '../../ui/theme';
import { useCanvas } from '../../viz/useCanvas';
import { href } from '../router';
import { loadPreview } from './previews';
import type { Preview } from './previews/types';
import styles from './LabCard.module.css';

/** Seconds one hover play takes at most. */
const CARD_PLAY = 3.2;

/** True once the element comes within `margin` of the viewport (then stays true). */
function useNearViewport(ref: React.RefObject<Element | null>, margin = '320px'): boolean {
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || near) return;
    if (typeof IntersectionObserver === 'undefined') {
      const id = requestAnimationFrame(() => setNear(true));
      return () => cancelAnimationFrame(id);
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setNear(true);
      },
      { rootMargin: margin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ref, margin, near]);
  return near;
}

export function LabCard({ lab, headingLevel = 4 }: { lab: LabEntry; headingLevel?: 3 | 4 }) {
  const cardRef = useRef<HTMLAnchorElement>(null);
  const near = useNearViewport(cardRef);
  const [preview, setPreview] = useState<Preview | null>(null);
  const colors = useChartColors();
  const reduced = usePrefersReducedMotion();
  const u = useRef(1);
  const raf = useRef(0);

  useEffect(() => {
    if (!near) return;
    let alive = true;
    void loadPreview(lab.id).then((p) => alive && setPreview(p));
    return () => {
      alive = false;
    };
  }, [near, lab.id]);

  const { canvasRef, redraw } = useCanvas((ctx, size) => {
    preview?.draw(ctx, size, colors, u.current, false);
  });

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const play = () => {
    void lab.preload?.();
    if (reduced || !preview || raf.current) return;
    const ms = Math.min(CARD_PLAY, preview.duration * 0.7) * 1000;
    let t0 = 0;
    u.current = 0;
    const tick = (now: number) => {
      if (!t0) t0 = now;
      u.current = Math.min(1, (now - t0) / ms);
      redraw();
      raf.current = u.current < 1 ? requestAnimationFrame(tick) : 0;
    };
    raf.current = requestAnimationFrame(tick);
  };

  const H = headingLevel === 3 ? 'h3' : 'h4';
  const status = lab.status === 'open' ? null : lab.status === 'preview' ? 'Preview' : 'Planned';
  return (
    <a
      ref={cardRef}
      className={styles.card}
      href={href(`/lab/${lab.id}`)}
      onPointerEnter={play}
      onFocus={play}
      data-status={lab.status}
    >
      <span className={styles.thumb} data-ready={preview ? '' : undefined}>
        <canvas ref={canvasRef} aria-hidden="true" />
      </span>
      <span className={styles.body}>
        <H className={styles.title}>{lab.title}</H>
        <span className={styles.pitch}>{lab.pitch}</span>
        <span className={styles.meta}>
          {int(lab.methodCount)} {lab.methodCount === 1 ? 'method' : 'methods'}
          {status && ` · ${status}`}
        </span>
      </span>
    </a>
  );
}
