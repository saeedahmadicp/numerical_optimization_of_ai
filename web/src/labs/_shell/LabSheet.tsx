/**
 * The phone controls sheet of LabShell (< 900 px): a modal bottom sheet at half height (drag up
 * or press the handle for full height). In its own chunk with the motion library, which it needs
 * for the drag gesture; it loads when the viewer first opens the controls.
 */
import { useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { IconButton } from '../../ui/components/Button';
import { useMotionTransitions } from '../../ui/motion';
import styles from './LabShell.module.css';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])';

export interface LabSheetProps {
  open: boolean;
  full: boolean;
  setFull: (f: boolean | ((f: boolean) => boolean)) => void;
  onClose: () => void;
  children: ReactNode;
}

export default function LabSheet({ open, full, setFull, onClose, children }: LabSheetProps) {
  const sheetRef = useRef<HTMLDivElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);
  const t = useMotionTransitions();

  // Focus the close button on open; Escape closes.
  useEffect(() => {
    if (!open) return;
    const id = requestAnimationFrame(() => closeBtn.current?.focus());
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      cancelAnimationFrame(id);
      window.removeEventListener('keydown', onKey);
    };
  }, [open, onClose]);

  // Keep Tab / Shift+Tab inside the sheet (the rest of the page is also `inert`).
  const trapTab = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Tab' || !sheetRef.current) return;
    const items = [...sheetRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
      (el) => el.getClientRects().length > 0,
    );
    if (items.length === 0) return;
    const first = items[0],
      last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="backdrop"
            className={styles.sheetBackdrop}
            data-full={full || undefined}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: t.exit }}
            transition={t.base}
            onClick={onClose}
          />
          <motion.div
            key="sheet"
            ref={sheetRef}
            role="dialog"
            aria-modal="true"
            aria-label="Controls"
            className={styles.sheet}
            data-full={full || undefined}
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%', transition: t.exit }}
            transition={t.slow}
            drag="y"
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={{ top: 0.15, bottom: 0.6 }}
            onDragEnd={(_, info) => {
              if (info.offset.y < -40) setFull(true);
              else if (info.offset.y > 120) {
                if (full) setFull(false);
                else onClose();
              }
            }}
            onKeyDown={trapTab}
          >
            <div className={styles.sheetHandle}>
              <button
                type="button"
                className={styles.grabber}
                aria-label={
                  full ? 'Shrink controls to half height' : 'Expand controls to full height'
                }
                aria-expanded={full}
                onClick={() => setFull((f) => !f)}
              >
                <span aria-hidden="true" />
              </button>
              <h2 className={styles.sheetTitle}>Controls</h2>
              <IconButton
                ref={closeBtn}
                icon="x"
                label="Close controls"
                onClick={onClose}
                tooltip={false}
              />
            </div>
            <div className={styles.railScroll}>{children}</div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
