import { IconButton } from '../ui/components/Button';
import styles from './viz.module.css';

export interface ViewToolbarProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  /** False while the view is the initial one: Reset stays in place, disabled (no layout change). */
  canReset: boolean;
  /** Reset's name (default "Reset view"). */
  resetLabel?: string;
  /** Optional follow toggle (the view tracks the current iterate), drawn after Reset. */
  follow?: { on: boolean; onToggle: () => void; label?: string };
  /** Accessible name of the toolbar (default "Plot view"). */
  label?: string;
  className?: string;
}

/**
 * The one view toolbar of every zoomable plot: top right, under the stage head; zoom in, zoom out,
 * reset (always shown, disabled until the view changes, so the bar never changes size), then an
 * optional follow toggle with its own icon (crosshair).
 */
export function ViewToolbar({
  onZoomIn,
  onZoomOut,
  onReset,
  canReset,
  resetLabel = 'Reset view',
  follow,
  label = 'Plot view',
  className,
}: ViewToolbarProps) {
  return (
    <div
      className={`${styles.viewToolbar} ${className ?? ''}`}
      role="toolbar"
      aria-label={label}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <IconButton size="sm" icon="plus" label="Zoom in" onClick={onZoomIn} tooltipSide="bottom" />
      <IconButton
        size="sm"
        icon="minus"
        label="Zoom out"
        onClick={onZoomOut}
        tooltipSide="bottom"
      />
      <IconButton
        size="sm"
        icon="reset"
        label={resetLabel}
        onClick={onReset}
        disabled={!canReset}
        tooltipSide="bottom"
      />
      {follow && (
        <IconButton
          size="sm"
          icon="crosshair"
          label={
            follow.label ??
            (follow.on ? 'Following the iterate (zoom or pan to stop)' : 'Follow the iterate')
          }
          aria-pressed={follow.on}
          data-on={follow.on || undefined}
          className={styles.viewToolbarToggle}
          onClick={follow.onToggle}
          tooltipSide="bottom"
        />
      )}
    </div>
  );
}
