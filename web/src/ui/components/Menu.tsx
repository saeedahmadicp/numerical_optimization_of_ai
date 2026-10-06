import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent as RKeyboardEvent,
  type ReactNode,
} from 'react';
import { useCopied } from '../copy';
import { Icon, type IconName } from './Icon';
import { IconButton } from './Button';
import styles from './Menu.module.css';

export type MenuItem =
  | { kind?: 'action'; label: string; icon?: IconName; onSelect: () => void }
  | { kind: 'copy'; label: string; icon?: IconName; text: string | (() => string) }
  | { kind: 'link'; label: string; icon?: IconName; href: string; external?: boolean };

export interface MenuProps {
  /** Accessible name of the trigger ("More actions for BFGS"). */
  label: string;
  items: readonly MenuItem[];
  icon?: IconName;
  /** Align the popup to the trigger's right edge (default) or left edge. */
  align?: 'right' | 'left';
  trigger?: ReactNode;
}

/**
 * An overflow menu (menu button pattern): ↑/↓/Home/End move, Enter/Space choose, Esc closes and
 * returns focus. Copy items confirm inline ("Copied") before the menu closes.
 */
export function Menu({ label, items, icon = 'more', align = 'right' }: MenuProps) {
  const [open, setOpen] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [, copy] = useCopied();
  const id = useId();
  const btn = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const closeTimer = useRef(0);

  useEffect(() => {
    if (!open) return;
    const first = list.current?.querySelector<HTMLElement>('[role="menuitem"]');
    first?.focus();
    const onDown = (e: PointerEvent) => {
      if (!list.current?.contains(e.target as Node) && !btn.current?.contains(e.target as Node))
        setOpen(false);
    };
    window.addEventListener('pointerdown', onDown);
    return () => window.removeEventListener('pointerdown', onDown);
  }, [open]);
  useEffect(() => () => window.clearTimeout(closeTimer.current), []);

  const close = (focus = true) => {
    setOpen(false);
    setCopiedIndex(null);
    if (focus) btn.current?.focus();
  };

  const onKey = (e: RKeyboardEvent<HTMLDivElement>) => {
    const els = [...(list.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])];
    const i = els.indexOf(document.activeElement as HTMLElement);
    let next = -1;
    if (e.key === 'ArrowDown') next = (i + 1) % els.length;
    else if (e.key === 'ArrowUp') next = (i - 1 + els.length) % els.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = els.length - 1;
    else if (e.key === 'Escape' || e.key === 'Tab') {
      if (e.key === 'Escape') e.preventDefault();
      close(e.key === 'Escape');
      return;
    }
    if (next >= 0) {
      e.preventDefault();
      els[next]?.focus();
    }
    e.stopPropagation();
  };

  const choose = async (item: MenuItem, index: number) => {
    if (item.kind === 'copy') {
      await copy(typeof item.text === 'function' ? item.text() : item.text);
      setCopiedIndex(index);
      window.clearTimeout(closeTimer.current);
      closeTimer.current = window.setTimeout(() => close(), 900);
      return;
    }
    if (item.kind !== 'link') item.onSelect();
    close(item.kind !== 'link');
  };

  return (
    <span className={styles.root}>
      <IconButton
        ref={btn}
        icon={icon}
        label={label}
        size="sm"
        tooltip={false}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        onClick={() => setOpen((o) => !o)}
        data-own-keys
      />
      {open && (
        <div
          ref={list}
          id={id}
          role="menu"
          aria-label={label}
          className={styles.menu}
          data-align={align}
          onKeyDown={onKey}
          data-own-keys
        >
          {items.map((item, i) => {
            const content = (
              <>
                <Icon
                  name={copiedIndex === i ? 'check' : (item.icon ?? defaultIcon(item))}
                  size={14}
                />
                <span>{copiedIndex === i ? 'Copied' : item.label}</span>
              </>
            );
            return item.kind === 'link' ? (
              <a
                key={i}
                role="menuitem"
                tabIndex={-1}
                className={styles.item}
                href={item.href}
                target={item.external ? '_blank' : undefined}
                rel={item.external ? 'noreferrer' : undefined}
                onClick={() => close(false)}
              >
                {content}
              </a>
            ) : (
              <button
                key={i}
                type="button"
                role="menuitem"
                tabIndex={-1}
                className={styles.item}
                onClick={() => void choose(item, i)}
              >
                {content}
              </button>
            );
          })}
        </div>
      )}
    </span>
  );
}

function defaultIcon(item: MenuItem): IconName {
  if (item.kind === 'copy') return 'copy';
  if (item.kind === 'link') return item.external ? 'external' : 'arrowRight';
  return 'chevronRight';
}
