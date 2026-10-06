import { useId, useRef, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react';
import { tabPanelProps } from './tabPanel';
import { motion } from 'motion/react';
import { useMotionTransitions } from '../motion';
import styles from './Controls.module.css';

export interface TabItem<T extends string> {
  id: T;
  label: ReactNode;
}

export interface TabsProps<T extends string> {
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  /** Prefix for tab/panel ids so panels can reference `${idBase}-panel-${id}`. */
  idBase?: string;
  className?: string;
}

/** WAI-ARIA tabs (automatic activation, roving tabindex). Render panels with `tabPanelProps`. */
export function Tabs<T extends string>({
  items,
  value,
  onChange,
  label,
  idBase,
  className,
}: TabsProps<T>) {
  const auto = useId();
  const motionT = useMotionTransitions();
  const base = idBase ?? auto;
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const idx = items.findIndex((t) => t.id === value);
  const onKeyDown = (e: KeyboardEvent) => {
    let next = -1;
    if (e.key === 'ArrowRight') next = (idx + 1) % items.length;
    else if (e.key === 'ArrowLeft') next = (idx - 1 + items.length) % items.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = items.length - 1;
    if (next < 0) return;
    e.preventDefault();
    e.stopPropagation();
    onChange(items[next].id);
    refs.current[next]?.focus();
  };
  return (
    <div
      role="tablist"
      aria-label={label}
      className={`${styles.tabs} ${className ?? ''}`}
      onKeyDown={onKeyDown}
    >
      {items.map((t, i) => {
        const sel = t.id === value;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            id={`${base}-tab-${t.id}`}
            type="button"
            role="tab"
            aria-selected={sel}
            aria-controls={`${base}-panel-${t.id}`}
            tabIndex={sel ? 0 : -1}
            className={styles.tab}
            onClick={() => onChange(t.id)}
          >
            {t.label}
            {sel && (
              <motion.span
                layoutId={`tabline-${base}`}
                className={styles.tabLine}
                transition={motionT.base}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

/**
 * The panel for one tab of a `<Tabs idBase=...>`: role, ids and labelling come from
 * `tabPanelProps`, so a lab never writes them by hand. Fills its container by default.
 * Set `focusable={false}` when the panel's content has its own tab stop (a scroll region, a table).
 */
export function TabPanel({
  idBase,
  id,
  children,
  className,
  style,
  focusable = true,
}: {
  idBase: string;
  id: string;
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  focusable?: boolean;
}) {
  return (
    <div
      {...tabPanelProps(idBase, id)}
      tabIndex={focusable ? 0 : -1}
      className={className}
      style={{ height: '100%', ...style }}
    >
      {children}
    </div>
  );
}
