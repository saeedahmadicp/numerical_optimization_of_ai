import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { IconButton } from '../ui/components/Button';
import { Icon } from '../ui/components/Icon';
import { Kbd } from '../ui/components/Misc';
import { useToast } from '../ui/components/Toast';
import { systemTheme, useTheme } from '../ui/theme';
import { href, useRoute } from './router';
import { Lockup } from './Logo';
import styles from './AppHeader.module.css';

const loadSearch = () => import('./Search');
const SearchDialog = lazy(loadSearch);

/**
 * Switches between the two themes, so every click changes what you see. Choosing the theme the
 * OS already uses stores "system" again, so the page goes back to following the OS.
 */
export function ThemeToggle() {
  const { mode, setPreference } = useTheme();
  const next = mode === 'dark' ? 'light' : 'dark';
  return (
    <IconButton
      icon={next === 'dark' ? 'moon' : 'sun'}
      label={`Switch to ${next} theme`}
      tooltipSide="bottom"
      onClick={() => setPreference(next === systemTheme() ? 'system' : next)}
    />
  );
}

export function ShareButton() {
  const toast = useToast();
  const share = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      toast('Link copied — it reproduces this view', 'check');
    } catch {
      toast('Copy the address bar to share this view', 'link');
    }
  };
  return (
    <IconButton icon="link" label="Copy link to this view" tooltipSide="bottom" onClick={share} />
  );
}

export interface Crumb {
  label: string;
  href?: string;
}

const NAV = [
  { id: 'labs', label: 'Labs', href: '/labs' },
  { id: 'methods', label: 'Methods', href: '/methods' },
  { id: 'research', label: 'Research', href: '/research' },
  { id: 'python', label: 'Python', href: '/python' },
] as const;

const typing = (el: EventTarget | null) =>
  el instanceof HTMLElement &&
  (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));

/** "/" and ⌘K / Ctrl K open the search palette. */
function useSearchShortcut(open: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        open();
      } else if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey && !typing(e.target)) {
        e.preventDefault();
        open();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);
}

function MobileMenu({ current }: { current: string }) {
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    panel.current?.querySelector<HTMLElement>('a')?.focus();
    const onDown = (e: PointerEvent) => {
      if (!panel.current?.contains(e.target as Node) && !btn.current?.contains(e.target as Node))
        setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        btn.current?.focus();
      }
    };
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);
  return (
    <div className={styles.mobileMenu}>
      <IconButton
        ref={btn}
        icon={open ? 'x' : 'menu'}
        label={open ? 'Close menu' : 'Menu'}
        tooltip={false}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      />
      {open && (
        <div ref={panel} className={styles.menuPanel}>
          <nav aria-label="Main">
            {NAV.map((n) => (
              <a
                key={n.id}
                href={href(n.href)}
                aria-current={current === n.id ? 'page' : undefined}
                onClick={() => setOpen(false)}
              >
                {n.label}
              </a>
            ))}
          </nav>
        </div>
      )}
    </div>
  );
}

export interface AppHeaderProps {
  crumbs?: Crumb[];
  actions?: ReactNode;
  /** Full-width (labs) instead of the 1240-px content column (pages). */
  wide?: boolean;
}

export function AppHeader({ crumbs = [], actions, wide = false }: AppHeaderProps) {
  const route = useRoute();
  const current =
    route.name === 'home' || route.name === 'labs' || route.name === 'lab' ? 'labs' : route.name;
  const [searching, setSearching] = useState(false);
  const openSearch = useCallback(() => setSearching(true), []);
  useSearchShortcut(openSearch);
  const trigger = useRef<HTMLButtonElement>(null);
  const closeSearch = () => {
    setSearching(false);
    requestAnimationFrame(() => trigger.current?.focus());
  };

  return (
    <header className={styles.header} data-wide={wide || undefined}>
      <div className={styles.inner}>
        <a className={styles.brand} href={href('/')} aria-label="numopt home">
          <Lockup size={19} />
        </a>
        {crumbs.length > 0 ? (
          <nav aria-label="Breadcrumb" className={styles.crumbs}>
            {crumbs.map((c, i) => (
              <span
                key={i}
                className={c.href ? styles.hideNarrow : undefined}
                style={{ display: 'contents' }}
              >
                <span
                  className={`${styles.crumbSep} ${c.href ? styles.hideNarrow : ''}`}
                  aria-hidden="true"
                >
                  /
                </span>
                {c.href ? (
                  <a className={styles.hideNarrow} href={c.href}>
                    {c.label}
                  </a>
                ) : (
                  <span className={styles.crumbCurrent} aria-current="page">
                    {c.label}
                  </span>
                )}
              </span>
            ))}
          </nav>
        ) : (
          <nav aria-label="Main" className={styles.nav}>
            {NAV.map((n) => (
              <a
                key={n.id}
                href={href(n.href)}
                aria-current={current === n.id ? 'page' : undefined}
              >
                {n.label}
              </a>
            ))}
          </nav>
        )}
        <div className={styles.spacer} />
        <button
          ref={trigger}
          type="button"
          className={styles.search}
          onClick={openSearch}
          onPointerEnter={() => void loadSearch()}
          onFocus={() => void loadSearch()}
          aria-keyshortcuts="/ Control+K Meta+K"
          aria-label="Search methods, problems and labs"
          aria-haspopup="dialog"
        >
          <Icon name="search" size={15} />
          <span className={styles.searchText}>Search methods, problems</span>
          <Kbd>/</Kbd>
        </button>
        <div className={styles.actions}>
          {actions}
          <ThemeToggle />
          {/* Every page, breadcrumbs or not: the only way to the sections on a phone. */}
          <MobileMenu current={current} />
        </div>
      </div>
      {searching && (
        <Suspense fallback={null}>
          <SearchDialog onClose={closeSearch} />
        </Suspense>
      )}
    </header>
  );
}
