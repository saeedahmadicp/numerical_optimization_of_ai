/**
 * The home hero figure: a live showcase that cycles through a few labs' signature scenes
 * (HERO_SCENES) — a descent race, root finding, the simplex walk, Runge's phenomenon. Each scene
 * plays its real runs once, holds the final frame, then crossfades to the next. The scene
 * switcher, a pause button and "Open this lab" sit under the figure; the caption states what was
 * run and how it stopped. Reduced motion: no cycling, each scene shows its final frame.
 *
 * No layout shift: the stage has a fixed aspect ratio, and the legends and captions of all scenes
 * are laid out in one grid cell each (inactive ones hidden), so the figure keeps one height across
 * scenes at every width. The reserved heights in the CSS cover the tallest scene, so the figure
 * also keeps its height while the later scenes load.
 *
 * Math: the titles and captions write their symbols as `$…$` TeX, typeset with KaTeX (brand.md §3);
 * the canvases' `ariaLabel`s keep plain text.
 *
 * Loading: the first scene shows as soon as it is built (Home.tsx starts its chunk, and KaTeX, at
 * boot, beside this one) and the web fonts and KaTeX are ready (at most FONT_WAIT ms; a caption
 * waits for KaTeX in any case, in its reserved box). The other scenes' chunks then download in parallel, and each is built in its
 * own idle period, well before the first crossfade; a scene that is not ready when its turn comes
 * holds the current final frame until it is.
 */
import { useEffect, useRef, useState } from 'react';
import { usePrefersReducedMotion } from '../../play/reducedMotion';
import { useChartColors } from '../../ui/theme';
import { Icon } from '../../ui/components/Icon';
import { Formula } from '../../ui/components/Formula';
import { Swatch } from '../../ui/components/MethodChip';
import { preloadKatex } from '../../ui/katex';
import { splitMath } from '../../ui/mathProse';
import { useCanvas } from '../../viz/useCanvas';
import { href } from '../router';
import { HERO_SCENES, SCENE_NAMES } from './heroScenes';
import { HeroSkeleton } from './HeroSkeleton';
import { loadPreview, prefetchPreview } from './previews';
import type { Preview } from './previews/types';
import styles from './HeroShowcase.module.css';

/** Seconds the final frame of a scene holds before the next scene. */
const HOLD = 1.9;
/** The hero plays each preview at this fraction of its nominal `duration` (owner: a brisker loop). */
const PACE = 0.72;

interface LayerProps {
  preview: Preview;
  active: boolean;
  /** Restart token: changes when the user picks a scene. */
  run: number;
  playing: boolean;
  reduced: boolean;
  onProgress: (u: number) => void;
  onDone: () => void;
}

function SceneLayer({ preview, active, run, playing, reduced, onProgress, onDone }: LayerProps) {
  const colors = useChartColors();
  const u = useRef(1);
  const elapsed = useRef(0);
  const cb = useRef({ onProgress, onDone });
  useEffect(() => {
    cb.current = { onProgress, onDone };
  });
  const { canvasRef, redraw } = useCanvas((ctx, size) =>
    preview.draw(ctx, size, colors, u.current, true),
  );

  // A new activation (or a restart) plays from the start.
  useEffect(() => {
    if (!active) return;
    elapsed.current = 0;
    u.current = reduced ? 1 : 0;
    cb.current.onProgress(u.current);
    redraw();
  }, [active, run, reduced, redraw]);

  useEffect(() => {
    if (!active || reduced || !playing) return;
    const play = preview.duration * PACE;
    let last = 0;
    let id = requestAnimationFrame(function tick(now) {
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now;
      elapsed.current += dt;
      const next = Math.min(1, elapsed.current / play);
      if (next !== u.current) {
        u.current = next;
        redraw();
      }
      cb.current.onProgress(Math.min(1, elapsed.current / (play + HOLD)));
      if (elapsed.current >= play + HOLD) {
        cb.current.onDone();
        return;
      }
      id = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(id);
  }, [active, run, playing, reduced, preview, redraw]);

  return (
    <canvas
      ref={canvasRef}
      className={styles.layer}
      data-active={active ? '' : undefined}
      role="img"
      aria-label={preview.ariaLabel}
      aria-hidden={active ? undefined : true}
    />
  );
}

/** Pauses when the figure scrolls out of view. */
function useInView(ref: React.RefObject<Element | null>): boolean {
  const [inView, setInView] = useState(true);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver((e) => setInView(e.some((x) => x.isIntersecting)), {
      threshold: 0.15,
    });
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);
  return inView;
}

/** A scene's state: built, still loading, or failed (left out of the cycle). */
type Slot = Preview | 'loading' | 'failed';

const ready = (s: Slot | undefined): s is Preview => typeof s === 'object';

/** The next scene after `from` that has not failed (`from` itself when all the others failed). */
function nextAfter(slots: Slot[], from: number): number {
  for (let k = 1; k <= slots.length; k++) {
    const j = (from + k) % slots.length;
    if (slots[j] !== 'failed') return j;
  }
  return from;
}

/** Keeps a numeric tuple or interval such as (−2.805, 3.131) or [2, 3] on one line: no-break
 *  spaces after its commas. */
const keepTuples = (text: string) =>
  text.replace(/[([](?:[−-]?[\d.]+, )+[−-]?[\d.]+[)\]]/g, (t) => t.replaceAll(', ', ',\u00a0'));

/** Keeps a joined name such as Newton–Raphson on one line: a word joiner after its en dash. */
const keepNames = (text: string) => text.replace(/(\p{L})–(\p{L})/gu, '$1–\u2060$2');

/** TeX that ends in a relation, as in `$\mathbf{x}^\star \approx$ (−2.805, 3.131)`. */
const ENDS_IN_RELATION = /(?:=|\\approx|\\le|\\ge)\s*$/;

/**
 * A title or caption: `$…$` parts are typeset with KaTeX (on one line each: see the CSS), the prose
 * between them keeps its numeric tuples and joined names on one line, and a value stays on the line
 * of the relation before it (`𝐱⋆ ≈` never ends a line). The captions render once KaTeX is in
 * (`mathReady`); if it fails to load, `Formula` shows a plain-text stand-in in the same place.
 */
function CaptionText({ text }: { text: string }) {
  const parts = splitMath(text);
  return parts.map((part, i) => {
    if (i % 2 === 1) return <Formula key={i} tex={part} fallback />;
    const prose = keepNames(keepTuples(part));
    return i > 0 && ENDS_IN_RELATION.test(parts[i - 1]) ? prose.replace(/^ /, '\u00a0') : prose;
  });
}

/** The longest the first scene waits for the web fonts and KaTeX, in ms. */
const FONT_WAIT = 1200;

/** The KaTeX faces the captions use: Main (upright, and bold for vectors) and Math (italic). */
const KATEX_FACES = ['400 1em KaTeX_Main', '700 1em KaTeX_Main', 'italic 400 1em KaTeX_Math'];

let math: Promise<void> | null = null;

/**
 * Settles when KaTeX and the faces in KATEX_FACES have loaded (or failed to load: then `Formula`
 * shows its plain-text stand-ins). Never rejects.
 */
function mathReady(): Promise<void> {
  math ??= preloadKatex()
    .then(() =>
      typeof document === 'undefined' || !document.fonts
        ? undefined
        : Promise.all(KATEX_FACES.map((f) => document.fonts.load(f))),
    )
    .then(
      () => undefined,
      () => undefined,
    );
  return math;
}

/**
 * Resolves when the page's web fonts, KaTeX and its fonts have loaded, or after FONT_WAIT ms. A
 * legend or caption first set in a fallback face would rewrap when the real faces arrive, and that
 * is a layout shift.
 */
function typeReady(): Promise<unknown> {
  if (typeof document === 'undefined' || !document.fonts) return Promise.resolve();
  return Promise.race([
    Promise.all([document.fonts.ready, mathReady()]),
    new Promise((resolve) => setTimeout(resolve, FONT_WAIT)),
  ]);
}

/** Runs `fn` when the main thread is idle (at the latest after `timeout` ms); returns a cancel. */
function whenIdle(fn: () => void, timeout = 1000): () => void {
  if (typeof requestIdleCallback === 'function') {
    const id = requestIdleCallback(fn, { timeout });
    return () => cancelIdleCallback(id);
  }
  const id = setTimeout(fn, 50);
  return () => clearTimeout(id);
}

export default function HeroShowcase() {
  const reduced = usePrefersReducedMotion();
  // One slot per HERO_SCENES entry. The first scene shows as soon as it is built; the others
  // are fetched in parallel and built one per idle period, long before the first crossfade.
  const [slots, setSlots] = useState<Slot[]>(() => HERO_SCENES.map(() => 'loading'));
  const [index, setIndex] = useState(0);
  /** The scene to show next, once it is built (the end of a play, or a pick of a loading scene). */
  const [target, setTarget] = useState<number | null>(null);
  const [run, setRun] = useState(0);
  const [paused, setPaused] = useState(false);
  const figRef = useRef<HTMLDivElement>(null);
  const bars = useRef<(HTMLSpanElement | null)[]>([]);
  const ensure = useRef<(i: number) => void>(() => {});
  const inView = useInView(figRef);
  // The captions are set once KaTeX and its fonts are in (nearly always before the first scene
  // shows: typeReady waits for them). On a slow network the figure shows after FONT_WAIT ms with
  // the caption block empty at its reserved height; the text then appears in place, typeset once,
  // instead of first in KaTeX's plain-text stand-ins and then rewrapped (a shift of the link).
  const [mathSet, setMathSet] = useState(false);
  useEffect(() => {
    let alive = true;
    void mathReady().then(() => alive && setMathSet(true));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;
    let cancel = () => {};
    const ensureScene = (i: number) =>
      loadPreview(HERO_SCENES[i]).then((p) => {
        if (!alive) return;
        setSlots((s) =>
          s[i] === 'loading' ? s.map((x, j) => (j === i ? (p ?? 'failed') : x)) : s,
        );
      });
    ensure.current = (i) => void ensureScene(i);
    // Home.tsx already started the first scene (and KaTeX) at boot; this joins the same promises.
    // The figure also waits for the web fonts and KaTeX (at most FONT_WAIT ms): a legend or caption
    // set in a fallback face would rewrap when the real faces arrive, and that is a layout shift.
    void Promise.all([loadPreview(HERO_SCENES[0]), typeReady()])
      .then(() => ensureScene(0))
      .then(() => {
        if (!alive) return;
        for (const id of HERO_SCENES.slice(1)) prefetchPreview(id);
        const build = (i: number) => {
          if (!alive || i >= HERO_SCENES.length) return;
          cancel = whenIdle(() => void ensureScene(i).then(() => build(i + 1)));
        };
        build(1);
      });
    return () => {
      alive = false;
      cancel();
    };
  }, []);

  // Show the requested scene once it is built. This adjusts state while rendering (React's
  // pattern for state that follows other state), so no effect and no extra frame.
  if (target !== null && slots[target] !== 'loading') {
    if (ready(slots[target])) {
      setIndex(target);
      if (target === index) setRun((r) => r + 1);
      setTarget(null);
    } else {
      // A failed target passes the turn to the next scene that has not failed.
      const j = nextAfter(slots, target);
      setTarget(j === index || slots[j] === 'failed' ? null : j);
    }
  } else if (target === null && slots[index] === 'failed') {
    // The first scene failed: start with the next one that builds.
    const j = nextAfter(slots, index);
    if (j !== index) setTarget(j);
  }

  if (!slots.some(ready)) return <HeroSkeleton />;

  const playing = !paused && inView;
  const pick = (i: number) => {
    if (ready(slots[i])) {
      setIndex(i);
      setRun((r) => r + 1);
      setTarget(null);
    } else {
      setTarget(i);
      ensure.current(i);
    }
  };
  const progress = (u: number) => {
    bars.current.forEach((b, i) => {
      if (b) b.style.transform = `scaleX(${i === index ? u : 0})`;
    });
  };
  const scenes = slots.flatMap((p, i) => (ready(p) ? [{ p, i }] : []));
  const shown = HERO_SCENES.flatMap((id, i) => (slots[i] === 'failed' ? [] : [{ id, i }]));

  return (
    <figure className={styles.figure}>
      <div className={styles.card}>
        <div className={styles.stage} ref={figRef}>
          {scenes.map(({ p, i }) => (
            <SceneLayer
              key={p.lab}
              preview={p}
              active={i === index}
              run={run}
              playing={playing}
              reduced={reduced}
              onProgress={progress}
              onDone={() => {
                // Hold the final frame until the next scene is built (it nearly always is).
                const j = nextAfter(slots, index);
                if (j !== index) setTarget(j);
              }}
            />
          ))}
        </div>
        {/* Every scene's legend sits in the same grid cell; only the active one is visible, so
            the block is always as tall as the tallest legend and never reflows between scenes. */}
        <div className={styles.legends}>
          {scenes.map(({ p, i }) => (
            <ul
              key={p.lab}
              className={styles.legend}
              aria-label="Methods"
              data-active={i === index ? '' : undefined}
              aria-hidden={i === index ? undefined : true}
            >
              {p.legend.map((l) => (
                <li key={l.label}>
                  <Swatch slot={l.slot} size={9} />
                  <span>{l.label}</span>
                  <span className={styles.note}>{l.note}</span>
                </li>
              ))}
            </ul>
          ))}
        </div>
      </div>

      <div className={styles.controls}>
        <div className={styles.scenes} role="group" aria-label="Scenes">
          {/* Every scene's button from the start (a scene still building plays when it is
              ready), so the row never changes while the later scenes load. */}
          {shown.map(({ id, i }) => (
            <button
              key={id}
              type="button"
              className={styles.scene}
              aria-pressed={i === index}
              onClick={() => pick(i)}
            >
              {SCENE_NAMES[id]}
              {!reduced && (
                <span className={styles.track} aria-hidden="true">
                  <span
                    className={styles.bar}
                    ref={(el) => {
                      bars.current[i] = el;
                    }}
                  />
                </span>
              )}
            </button>
          ))}
        </div>
        {!reduced && (
          <button
            type="button"
            className={styles.pause}
            onClick={() => setPaused((v) => !v)}
            aria-label={paused ? 'Play the showcase' : 'Pause the showcase'}
            title={paused ? 'Play' : 'Pause'}
          >
            <Icon name={paused ? 'play' : 'pause'} size={14} />
          </button>
        )}
      </div>

      {/* Same for the captions: all stacked, the tallest sets the height. */}
      <figcaption className={styles.captions}>
        {mathSet &&
          scenes.map(({ p, i }) => (
            <span
              key={p.lab}
              className={styles.caption}
              data-active={i === index ? '' : undefined}
              aria-hidden={i === index ? undefined : true}
            >
              <span className={styles.captionTitle}>
                <CaptionText text={p.title} />.
              </span>{' '}
              <CaptionText text={p.caption} />{' '}
              <a
                className={styles.open}
                href={href(`/lab/${p.lab}`)}
                tabIndex={i === index ? undefined : -1}
              >
                Open this lab
                <Icon name="arrowRight" size={13} />
              </a>
            </span>
          ))}
      </figcaption>
    </figure>
  );
}
