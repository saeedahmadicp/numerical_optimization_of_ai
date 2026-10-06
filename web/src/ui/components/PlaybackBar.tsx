import { useRef, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import type { TracePlayer } from '../../play/useTracePlayer';
import { SPEEDS } from '../../play/timeline';
import { int } from '../../core/format';
import { seriesVar } from '../colors';
import { IconButton } from './Button';
import { Tooltip } from './Tooltip';
import styles from './PlaybackBar.module.css';

export interface PlaybackBarProps {
  player: TracePlayer;
  /** Series slots of the compared methods (draws a tick where each one ends). */
  slots?: number[];
  /**
   * The step the readout names (default `player.k`). A lab that cross-fades between steps can
   * switch the number halfway through the fade, with its card and figure.
   */
  displayK?: number;
  /**
   * What the readout counts at the playhead, when it is not the trace index: e.g. the update
   * count of a stochastic method whose trace records every 3rd update ("update 111 / 800").
   * The slider's value text says the same.
   */
  readout?: { label: string; value: number; total: number };
  className?: string;
}

const fmtSpeed = (s: number) => (s < 1 ? `${s}×`.replace('0.', '.') : `${s}×`);

/** Transport controls + scrubber for a TracePlayer. Keyboard: see usePlayerKeyboard. */
const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);

export function PlaybackBar({ player, slots, className, displayK, readout }: PlaybackBarProps) {
  const shownK = displayK ?? player.k;
  const label = readout?.label ?? 'k';
  const value = readout?.value ?? shownK;
  const total = readout?.total ?? player.maxK;
  // The current count keeps the width of the largest one, so "k" never slides during playback.
  const digits = int(total).length;
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const { maxK } = player;
  // Progress in [0, 1]. The head and the fill move by transform only (compositor, no layout).
  const p = maxK > 0 ? Math.min(1, Math.max(0, player.t / maxK)) : 0;

  // Pointer → step, over the inset track (the head's center never leaves the track's ends).
  const fromClient = (x: number) => {
    const r = track.current?.getBoundingClientRect();
    if (!r || r.width === 0) return;
    const u = Math.min(1, Math.max(0, (x - r.left) / r.width));
    player.seek(Math.round(u * maxK));
  };
  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragging.current = true;
    player.pause();
    fromClient(e.clientX);
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => dragging.current && fromClient(e.clientX);
  const onUp = () => {
    dragging.current = false;
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const map: Record<string, () => void> = {
      ArrowRight: () => player.step(e.shiftKey ? 10 : 1),
      ArrowUp: () => player.step(1),
      ArrowLeft: () => player.step(e.shiftKey ? -10 : -1),
      ArrowDown: () => player.step(-1),
      PageUp: () => player.step(10),
      PageDown: () => player.step(-10),
      Home: () => player.reset(),
      End: () => player.toEnd(),
    };
    const fn = map[e.key];
    if (fn) {
      e.preventDefault();
      e.stopPropagation();
      fn();
    }
  };
  const nextSpeed = () => {
    const i = SPEEDS.indexOf(player.speed as (typeof SPEEDS)[number]);
    player.setSpeed(SPEEDS[(i + 1) % SPEEDS.length]);
  };

  return (
    <div className={`${styles.bar} ${className ?? ''}`} role="group" aria-label="Playback">
      <div className={styles.buttons}>
        <IconButton
          className={styles.hideNarrow}
          icon="reset"
          label="Restart"
          shortcut="R"
          onClick={player.reset}
          size="sm"
        />
        <IconButton
          icon="stepBack"
          label="Previous step"
          shortcut="←"
          onClick={() => player.step(-1)}
          disabled={player.k <= 0 && player.t <= 0}
          size="sm"
        />
        <IconButton
          className={styles.play}
          variant="secondary"
          icon={player.playing ? 'pause' : 'play'}
          label={player.playing ? 'Pause' : 'Play'}
          shortcut="Space"
          onClick={player.toggle}
        />
        <IconButton
          icon="stepForward"
          label="Next step"
          shortcut="→"
          onClick={() => player.step(1)}
          disabled={player.atEnd}
          size="sm"
        />
        <IconButton
          className={styles.hideNarrow}
          icon="skipEnd"
          label="Last step"
          shortcut="End"
          onClick={player.toEnd}
          size="sm"
        />
      </div>
      <div
        className={styles.scrub}
        role="slider"
        tabIndex={0}
        aria-label={readout ? `Iteration (${readout.label})` : 'Iteration'}
        aria-valuemin={0}
        aria-valuemax={maxK}
        aria-valuenow={shownK}
        aria-valuetext={
          readout
            ? `${cap(readout.label)} ${int(value)} of ${int(total)}`
            : `Iteration ${shownK} of ${maxK}`
        }
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onKeyDown={onKey}
      >
        <div ref={track} className={styles.track} style={{ ['--p' as string]: p } as CSSProperties}>
          <div className={styles.rail} />
          <div className={styles.progress} />
          {slots &&
            maxK > 0 &&
            player.lengths.map((len, i) =>
              len - 1 < maxK ? (
                <span
                  key={i}
                  className={styles.endMark}
                  style={{
                    left: `${((len - 1) / maxK) * 100}%`,
                    ['--_c' as string]: seriesVar(slots[i] ?? i),
                  }}
                />
              ) : null,
            )}
          <div className={styles.headTrack}>
            <div className={styles.head} />
          </div>
        </div>
      </div>
      <div className={styles.readout} aria-live="off">
        <span className={styles.hideNarrow}>{label}</span>
        <span className={styles.k} style={{ minWidth: `${digits}ch` }}>
          {int(value)}
        </span>
        <span>/ {int(total)}</span>
      </div>
      <Tooltip content="Playback speed" shortcut="+ / −">
        <button
          type="button"
          className={styles.speed}
          onClick={nextSpeed}
          aria-label={`Speed ${player.speed} times`}
        >
          {fmtSpeed(player.speed)}
        </button>
      </Tooltip>
    </div>
  );
}
