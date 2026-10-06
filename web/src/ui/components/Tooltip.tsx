import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  cloneElement,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from 'react';
import styles from './Tooltip.module.css';

export interface TooltipProps {
  content: ReactNode;
  shortcut?: string;
  side?: 'top' | 'bottom';
  /**
   * Link the tip to the trigger with aria-describedby (default true). Turn it off when the tip
   * only repeats the trigger's accessible name (icon buttons), so screen readers do not read it twice.
   */
  describe?: boolean;
  /**
   * A toggletip (help buttons): a click or tap also opens it and keeps it open until the next
   * click, a pointerdown outside or Escape. Touch devices have no hover, so help needs this.
   */
  toggle?: boolean;
  children: ReactElement<{ 'aria-describedby'?: string; 'aria-expanded'?: boolean }>;
}

/**
 * Hover/focus tooltip. The trigger keeps its own accessible name; the tip is a description.
 * The tip is only mounted while open (so it never widens the page), is nudged horizontally to
 * stay inside the viewport and flips to the other side when it would leave the top or bottom.
 */
export function Tooltip({
  content,
  shortcut,
  side = 'top',
  describe = true,
  toggle = false,
  children,
}: TooltipProps) {
  const id = useId();
  const [hovered, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const open = hovered || pinned;
  const wrapRef = useRef<HTMLSpanElement>(null);
  const [shift, setShift] = useState(0);
  const [flip, setFlip] = useState(false);
  const tip = useRef<HTMLSpanElement>(null);
  const show = () => setOpen(true);
  const hide = () => {
    setOpen(false);
    setShift(0);
    setFlip(false);
  };
  const close = () => {
    hide();
    setPinned(false);
  };
  // A pinned toggletip closes on a pointerdown anywhere else.
  useEffect(() => {
    if (!pinned) return;
    const onDown = (e: PointerEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) {
        setPinned(false);
        setOpen(false);
      }
    };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [pinned]);
  useLayoutEffect(() => {
    if (!open || !tip.current) return;
    const r = tip.current.getBoundingClientRect();
    const margin = 8;
    const dx =
      r.right > window.innerWidth - margin
        ? window.innerWidth - margin - r.right
        : r.left < margin
          ? margin - r.left
          : 0;
    if (dx !== 0) setShift(dx);
    if (side === 'top' ? r.top < margin : r.bottom > window.innerHeight - margin) setFlip(true);
  }, [open, side]);
  const below = (side === 'bottom') !== flip;
  return (
    <span
      ref={wrapRef}
      className={`${styles.wrap} ${below ? styles.bottom : ''}`}
      onPointerEnter={(e) => e.pointerType === 'mouse' && show()}
      onPointerLeave={() => !pinned && hide()}
      onFocus={(e) => e.target.matches(':focus-visible') && show()}
      onBlur={() => !pinned && hide()}
      onKeyDown={(e) => e.key === 'Escape' && close()}
      onClick={
        toggle
          ? () => {
              if (pinned) close();
              else setPinned(true);
            }
          : undefined
      }
    >
      {isValidElement(children)
        ? cloneElement(children, {
            'aria-describedby': open && describe ? id : undefined,
            ...(toggle ? { 'aria-expanded': open } : {}),
          })
        : children}
      {open && (
        <span
          ref={tip}
          role={describe ? 'tooltip' : undefined}
          aria-hidden={describe ? undefined : true}
          id={id}
          className={`${styles.tip} ${toggle ? styles.long : ''}`}
          style={{ marginLeft: shift }}
        >
          {content}
          {shortcut && <span className={styles.kbd}>{shortcut}</span>}
        </span>
      )}
    </span>
  );
}
