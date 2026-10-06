/**
 * The shared lab layout (LabShell, RailSection).
 *
 * Rail section order — one rule for every lab, top to bottom:
 *
 *   1. Try this           (`presets`; LabShell renders it first)
 *   2. Problem            (ProblemPicker: the picker, the formula, the description)
 *   3. Problem data       (inputs that define the problem instance: bracket, interval, point,
 *                          nodes, objective, cities, search direction)
 *   4. Methods            (MethodSlots)
 *   5. Shared run settings (settings every method shares: step sweep, sampling, random seed —
 *                          SeedControl renders its own "Random seed" section)
 *   6. Start point        (StartPointFields; last, with the shared "Click the plot" hint)
 *
 * A lab leaves out the sections it does not have and never reorders the rest.
 */
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import { AppHeader, ShareButton } from '../../app/AppHeader';
import { href } from '../../app/router';
import { Button } from '../../ui/components/Button';
import { PLAYER_SCOPE_ATTR } from '../../play/useTracePlayer';
import type { LabEntry } from '../index';
import type { LabPreset } from './labState';
import { TryThis } from './presets';
import { FocusPicker, type FocusRun } from './blocks';
import styles from './LabShell.module.css';

export interface Insight {
  id: string;
  title: ReactNode;
  /** Accessible name of the panel when `title` is not a string (e.g. a tab list). */
  label?: string;
  content: ReactNode;
  actions?: ReactNode;
  /** Take the remaining height in the side column. */
  grow?: boolean;
  /** Span the full width in the two-column (tablet) layout. */
  wide?: boolean;
  /** Fixed height in the side column (px). */
  height?: number;
}

export interface LabShellProps {
  lab: Pick<LabEntry, 'id' | 'title' | 'pitch'>;
  /** Left rail content: compose RailSection, ProblemPicker, MethodSlots, StartPointFields... */
  controls: ReactNode;
  /** "Try this" presets, shown as the first rail section (see presets.tsx). */
  presets?: readonly LabPreset[];
  /** Lab URL keys every preset removes (e.g. moved cities `c`, capacity `C`). */
  presetClearKeys?: readonly string[];
  /**
   * A one-line hint over the plot's bottom-left corner, usually
   * `<StartPointHint variable="𝐱₀" />` ("Click the plot to set 𝐱₀"). It overlays the plot (no
   * layout shift), shows on phones too, and fades out after the first click on the stage.
   */
  stageHint?: ReactNode;
  /** A centered note over the stage (empty state: "Add a method to compare — …"). */
  stageNotice?: ReactNode;
  /** A deferred run is computing: dim the stage and say "Recomputing…" (useLabRunsState). */
  pending?: boolean;
  /** Left side of the stage header (problem name, formula, result chips). */
  stageTitle?: ReactNode;
  /** Right side of the stage header (view toggles). */
  stageToolbar?: ReactNode;
  /** The main visualization. Fills the stage. Mark its focus target with `data-plot-focus`. */
  stage: ReactNode;
  /** Docked under the stage — usually <PlaybackBar>. */
  playback?: ReactNode;
  insights?: Insight[];
  /**
   * Phones and tablets (< 900 px): the stage height as a CSS length, when the visualization has
   * a natural aspect (e.g. `calc(50cqw + 44px)` for two square panels side by side; `cqw` is the
   * stage card's width). Default `min(78vw, 62dvh)`, at least 300 px.
   */
  stageHeightNarrow?: string;
  /**
   * The method shown in detail: one swatch picker in the header of the details insight (the one
   * with `grow`, or `focus.insight`), the same place in every lab. Stage panels that follow the
   * focus read the same state and do not render a second picker. Hidden with one run.
   */
  focus?: {
    runs: readonly FocusRun[];
    value: string;
    onChange: (value: string) => void;
    /** Insight id that shows the picker (default: the `grow` insight, else the last one). */
    insight?: string;
  };
}

// The phone sheet (and the motion library its drag gesture needs) loads on first open.
const LabSheet = lazy(() => import('./LabSheet'));

/**
 * The shared lab layout: header, control rail, stage + playback, insights column.
 * ≥1280px: three columns. 900–1279px: rail + stage, insights below across the full width.
 * <900px: stage first, controls in a modal bottom sheet (half height, so the plot stays visible;
 * drag up or press the handle for full height).
 */
export function LabShell({
  lab,
  controls: railControls,
  presets,
  presetClearKeys,
  stageNotice,
  stageHint,
  pending = false,
  stageTitle,
  stageToolbar,
  stage,
  playback,
  insights = [],
  stageHeightNarrow,
  focus,
}: LabShellProps) {
  const controls = (
    <>
      {presets && presets.length > 0 && (
        <RailSection title="Try this">
          <TryThis presets={presets} clearKeys={presetClearKeys} />
        </RailSection>
      )}
      {railControls}
    </>
  );
  const [sheet, setSheet] = useState(false);
  const [sheetLoaded, setSheetLoaded] = useState(false);
  const [full, setFull] = useState(false);
  const [hintSeen, setHintSeen] = useState(false);
  const fab = useRef<HTMLButtonElement>(null);
  const stageRef = useRef<HTMLElement>(null);
  const wasOpen = useRef(false);

  const openSheet = () => {
    setFull(false);
    setSheetLoaded(true);
    setSheet(true);
    // Bring the plot into view above the half-height sheet so parameter changes are visible.
    window.scrollTo({ top: 0 });
  };
  const closeSheet = useCallback(() => setSheet(false), []);

  // Return focus to the Controls button when the sheet closes (the sheet focuses its own close
  // button when it opens).
  useEffect(() => {
    if (sheet) {
      wasOpen.current = true;
      return;
    }
    if (wasOpen.current) {
      wasOpen.current = false;
      fab.current?.focus();
    }
  }, [sheet]);

  const skipToStage = () => {
    const target =
      stageRef.current?.querySelector<HTMLElement>('[data-plot-focus]') ?? stageRef.current;
    target?.focus();
  };

  const focusRuns = focus && focus.runs.length > 1 ? focus : null;
  const focusInsight =
    focus?.insight ?? insights.find((i) => i.grow)?.id ?? insights[insights.length - 1]?.id;

  return (
    <div className={styles.shell}>
      <div className={styles.page} inert={sheet || undefined}>
        <button type="button" className="skip-link" onClick={skipToStage}>
          Skip to visualization
        </button>
        <AppHeader
          wide
          crumbs={[{ label: 'Labs', href: href('/labs') }, { label: lab.title }]}
          actions={<ShareButton />}
        />
        <div className={styles.body}>
          <aside className={styles.rail} aria-label="Controls">
            <div className={styles.railScroll}>
              <div className={styles.railIntro}>
                <h1 className={styles.railTitle}>{lab.title}</h1>
                <p className={styles.railPitch}>{lab.pitch}</p>
              </div>
              {!sheet && controls}
            </div>
          </aside>

          <main className={styles.main}>
            {/* The page heading and pitch on phones and small tablets, where the rail is a sheet. */}
            <div className={styles.mobileIntro}>
              <h1 className={styles.mobileTitle}>{lab.title}</h1>
              <p className={styles.mobilePitch}>{lab.pitch}</p>
            </div>
            <section
              ref={stageRef}
              className={styles.stageCard}
              aria-label="Visualization"
              tabIndex={-1}
              {...{ [PLAYER_SCOPE_ATTR]: '' }}
            >
              {(stageTitle || stageToolbar) && (
                <div className={styles.stageHead}>
                  <div className={styles.stageTitle}>{stageTitle}</div>
                  {stageToolbar && <div className={styles.stageToolbar}>{stageToolbar}</div>}
                </div>
              )}
              <div
                className={styles.stage}
                data-pending={pending || undefined}
                onPointerDownCapture={stageHint && !hintSeen ? () => setHintSeen(true) : undefined}
                style={
                  stageHeightNarrow
                    ? ({
                        '--stage-h-narrow': stageHeightNarrow,
                        '--stage-min-narrow': '200px',
                      } as CSSProperties)
                    : undefined
                }
              >
                {stage}
                {/* An overlay, so the stage head keeps its size while a run recomputes. */}
                <span className={styles.pending} role="status" data-on={pending || undefined}>
                  {pending ? 'Recomputing…' : ''}
                </span>
                {stageHint && (
                  <p className={styles.stageHint} data-seen={hintSeen || undefined}>
                    {stageHint}
                  </p>
                )}
                {stageNotice && (
                  <div className={styles.stageNotice} role="note">
                    <div>{stageNotice}</div>
                  </div>
                )}
              </div>
              {playback && (
                <div className={styles.stageFoot}>
                  <div className={styles.stageFootMain}>{playback}</div>
                  {/* Phones: the Controls button docks into the playback bar (never covers the chart). */}
                  {!sheet && (
                    <Button
                      ref={fab}
                      className={styles.dockControls}
                      variant="primary"
                      size="sm"
                      icon="sliders"
                      onClick={openSheet}
                      aria-haspopup="dialog"
                      aria-label="Controls"
                    >
                      <span className={styles.dockLabel}>Controls</span>
                    </Button>
                  )}
                </div>
              )}
            </section>
          </main>

          {insights.length > 0 && (
            <div className={styles.insights}>
              {insights.map((ins) => (
                <section
                  key={ins.id}
                  className={styles.insight}
                  data-grow={ins.grow || undefined}
                  data-wide={ins.wide || undefined}
                  style={ins.height ? { flex: `0 0 ${ins.height}px` } : undefined}
                  aria-label={ins.label ?? (typeof ins.title === 'string' ? ins.title : ins.id)}
                >
                  <header className={styles.insightHead}>
                    {typeof ins.title === 'string' ? (
                      <h2 className={styles.insightTitle}>{ins.title}</h2>
                    ) : (
                      ins.title
                    )}
                    {(ins.actions || (focusRuns && ins.id === focusInsight)) && (
                      <div className={styles.insightActions}>
                        {focusRuns && ins.id === focusInsight && (
                          <FocusPicker
                            runs={focusRuns.runs}
                            value={focusRuns.value}
                            onChange={focusRuns.onChange}
                          />
                        )}
                        {ins.actions}
                      </div>
                    )}
                  </header>
                  <div className={styles.insightBody}>{ins.content}</div>
                </section>
              ))}
            </div>
          )}
        </div>

        {!sheet && !playback && (
          <Button
            ref={fab}
            className={styles.fab}
            variant="primary"
            size="lg"
            icon="sliders"
            onClick={openSheet}
            aria-haspopup="dialog"
          >
            Controls
          </Button>
        )}
      </div>

      {sheetLoaded && (
        <Suspense fallback={null}>
          <LabSheet open={sheet} full={full} setFull={setFull} onClose={closeSheet}>
            {controls}
          </LabSheet>
        </Suspense>
      )}
    </div>
  );
}

export function RailSection({
  title,
  actions,
  children,
}: {
  title: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHead}>
        {typeof title === 'string' ? <h2 className={styles.sectionTitle}>{title}</h2> : title}
        {actions}
      </div>
      {children}
    </section>
  );
}
