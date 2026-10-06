import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Icon } from './Icon';
import styles from './Select.module.css';
import { useMotionTransitions } from '../motion';

export interface SelectOption<T extends string = string> {
  value: T;
  label: string;
  description?: string;
  /** Leading visual (swatch, icon). */
  leading?: ReactNode;
  group?: string;
  disabled?: boolean;
  /** Extra text matched by the search box. */
  keywords?: string;
}

export interface SelectProps<T extends string = string> {
  value: T | null;
  options: SelectOption<T>[];
  onChange: (v: T) => void;
  label: string;
  placeholder?: string;
  /** Show a search box (default: when there are more than 6 options). */
  searchable?: boolean;
  /** Custom trigger content (defaults to the selected label). */
  renderValue?: (o: SelectOption<T> | undefined) => ReactNode;
  className?: string;
  /** Close and keep focus on the trigger after choosing (default true). */
  closeOnSelect?: boolean;
}

/** Searchable select (ARIA combobox + listbox). Type to filter, ↑/↓ to move, Enter to choose. */
export function Select<T extends string = string>({
  value,
  options,
  onChange,
  label,
  placeholder = 'Select…',
  searchable,
  renderValue,
  className,
  closeOnSelect = true,
}: SelectProps<T>) {
  const id = useId();
  const motionT = useMotionTransitions();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const showSearch = searchable ?? options.length > 6;
  /** WAI-ARIA listbox type-ahead (no search box): characters typed within 500 ms. */
  const typed = useRef({ text: '', at: 0 });

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) =>
      `${o.label} ${o.description ?? ''} ${o.keywords ?? ''} ${o.value}`.toLowerCase().includes(q),
    );
  }, [options, query]);

  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: PointerEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', onDoc);
    return () => document.removeEventListener('pointerdown', onDoc);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const i = Math.max(
      0,
      filtered.findIndex((o) => o.value === value),
    );
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setActive(i);
    requestAnimationFrame(() => (showSearch ? input.current : listRef.current)?.focus());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: 'nearest' });
  }, [active]);

  const choose = (o: SelectOption<T> | undefined) => {
    if (!o || o.disabled) return;
    onChange(o.value);
    if (closeOnSelect) {
      setOpen(false);
      setQuery('');
      trigger.current?.focus();
    }
  };

  const onKeyDown = (e: KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((a) => Math.min(filtered.length - 1, a + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (e.key === 'Home' && !showSearch) {
      setActive(0);
    } else if (e.key === 'End' && !showSearch) {
      setActive(filtered.length - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      choose(filtered[active]);
    } else if (
      !showSearch &&
      e.key.length === 1 &&
      e.key !== ' ' &&
      !e.ctrlKey &&
      !e.metaKey &&
      !e.altKey
    ) {
      // Type-ahead: jump to the next option whose label starts with the typed characters.
      const now = e.timeStamp || performance.now();
      const t = typed.current;
      t.text = now - t.at > 500 ? e.key.toLowerCase() : t.text + e.key.toLowerCase();
      t.at = now;
      const same = t.text.split('').every((c) => c === t.text[0]);
      const prefix = same ? t.text[0] : t.text;
      const n = filtered.length;
      const startAt = same || t.text.length === 1 ? active + 1 : active;
      for (let i = 0; i < n; i++) {
        const j = (startAt + i) % n;
        if (!filtered[j].disabled && filtered[j].label.toLowerCase().startsWith(prefix)) {
          setActive(j);
          break;
        }
      }
    } else if (e.key === 'Escape' || e.key === 'Tab') {
      if (e.key === 'Escape') e.preventDefault();
      setOpen(false);
      setQuery('');
      if (e.key === 'Escape') trigger.current?.focus();
    }
  };

  const listId = `${id}-list`;
  const optId = (i: number) => `${id}-opt-${i}`;

  return (
    <div ref={root} className={`${styles.root} ${className ?? ''}`}>
      <button
        ref={trigger}
        type="button"
        className={styles.trigger}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-label={`${label}: ${selected?.label ?? placeholder}`}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            e.stopPropagation();
            setOpen(true);
          }
        }}
      >
        <span className={styles.triggerLabel}>
          {renderValue ? (
            renderValue(selected)
          ) : selected ? (
            selected.label
          ) : (
            <span className={styles.placeholder}>{placeholder}</span>
          )}
        </span>
        <Icon name="chevronDown" className={styles.chev} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            className={styles.popover}
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, transition: motionT.exit }}
            transition={motionT.fast}
          >
            {showSearch && (
              <div className={styles.search}>
                <Icon name="search" />
                <input
                  ref={input}
                  className={styles.searchInput}
                  role="combobox"
                  aria-label={`Search ${label}`}
                  aria-expanded="true"
                  aria-controls={listId}
                  aria-autocomplete="list"
                  aria-activedescendant={filtered[active] ? optId(active) : undefined}
                  placeholder="Search…"
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setActive(0);
                  }}
                  onKeyDown={onKeyDown}
                />
              </div>
            )}
            <ul
              ref={listRef}
              id={listId}
              role="listbox"
              aria-label={label}
              tabIndex={showSearch ? -1 : 0}
              aria-activedescendant={!showSearch && filtered[active] ? optId(active) : undefined}
              className={styles.list}
              onKeyDown={showSearch ? undefined : onKeyDown}
            >
              {filtered.length === 0 && <li className={styles.empty}>No matches</li>}
              {filtered.map((o, i) => {
                const header =
                  o.group && (i === 0 || filtered[i - 1].group !== o.group) ? o.group : null;
                return (
                  <li key={o.value} role="presentation">
                    {header && (
                      <div className={styles.groupLabel} role="presentation">
                        {header}
                      </div>
                    )}
                    <div
                      id={optId(i)}
                      role="option"
                      data-index={i}
                      aria-selected={o.value === value}
                      aria-disabled={o.disabled || undefined}
                      data-active={i === active}
                      className={styles.option}
                      onPointerMove={() => setActive(i)}
                      onClick={() => choose(o)}
                    >
                      {o.leading}
                      <span className={styles.optionText}>
                        <span>{o.label}</span>
                        {o.description && (
                          <span className={styles.optionDesc}>{o.description}</span>
                        )}
                      </span>
                      {o.value === value && <Icon name="check" className={styles.check} />}
                    </div>
                  </li>
                );
              })}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export const Combobox = Select;
