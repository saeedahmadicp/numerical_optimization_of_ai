/**
 * The search palette (`/` or ⌘K / Ctrl K): labs, methods and problems from the Python registry.
 * Choosing a method opens its lab with the method preselected; a problem opens its lab with the
 * problem selected. Loaded lazily (with the catalog index) the first time it opens.
 *
 * A native modal <dialog> (focus trap, Esc, inert page) with the ARIA combobox + listbox pattern.
 */
import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { LABS } from '../labs';
import { Formula } from '../ui/components/Formula';
import { Icon } from '../ui/components/Icon';
import { Kbd } from '../ui/components/Misc';
import { loadCatalogIndex } from './catalog';
import type { CatalogIndex } from './catalogTypes';
import { buildSearchItems, searchItems, type SearchItem, type SearchKind } from './searchIndex';
import { navigate } from './router';
import styles from './Search.module.css';

const GROUP: Record<SearchKind, string> = { lab: 'Labs', method: 'Methods', problem: 'Problems' };

export default function SearchDialog({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const listId = useId();
  const [index, setIndex] = useState<CatalogIndex | null>(null);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);

  useEffect(() => {
    const d = dialog.current;
    if (d && !d.open) d.showModal();
    input.current?.focus();
    let alive = true;
    loadCatalogIndex().then((ix) => alive && setIndex(ix));
    return () => {
      alive = false;
    };
  }, []);

  const items = useMemo(
    () => buildSearchItems(LABS, index ?? { methods: [], problems: [], research: [] }),
    [index],
  );
  const results = useMemo(() => searchItems(items, query), [items, query]);
  const [lastQuery, setLastQuery] = useState(query);
  if (lastQuery !== query) {
    setLastQuery(query);
    setActive(0);
  }
  const current = results[Math.min(active, results.length - 1)];

  // The result count, announced politely ~300 ms after typing stops (WCAG 4.1.3).
  const [announce, setAnnounce] = useState('');
  useEffect(() => {
    if (!index) return;
    const id = window.setTimeout(() => setAnnounce(describeCounts(results)), 300);
    return () => window.clearTimeout(id);
  }, [results, index]);
  const optId = (i: number) => `${listId}-o${i}`;

  const go = (item: SearchItem | undefined) => {
    if (!item) return;
    onClose();
    navigate(item.href.slice(1));
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const n = results.length;
      if (!n) return;
      setActive((a) => (e.key === 'ArrowDown' ? (a + 1) % n : (a - 1 + n) % n));
    } else if (e.key === 'Home' && e.ctrlKey) setActive(0);
    else if (e.key === 'Enter') {
      e.preventDefault();
      go(current);
    }
    e.stopPropagation();
  };

  // Keep the active option in view.
  useEffect(() => {
    document.getElementById(optId(active))?.scrollIntoView({ block: 'nearest' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, results]);

  return (
    <dialog
      ref={dialog}
      className={styles.dialog}
      aria-label="Search labs, methods and problems"
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === dialog.current) onClose();
      }}
    >
      <div className={styles.panel}>
        <div className={styles.field}>
          <Icon name="search" size={18} />
          <input
            ref={input}
            className={styles.input}
            role="combobox"
            aria-expanded="true"
            aria-controls={listId}
            aria-autocomplete="list"
            aria-activedescendant={current ? optId(results.indexOf(current)) : undefined}
            aria-label="Search"
            placeholder="Search labs, methods, problems…"
            value={query}
            spellCheck={false}
            autoComplete="off"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKey}
          />
          <Kbd>Esc</Kbd>
        </div>
        <div role="status" className="visually-hidden">
          {announce}
        </div>
        <div id={listId} role="listbox" aria-label="Results" className={styles.list}>
          {results.length === 0 && (
            <p className={styles.empty}>
              {index ? `Nothing in the catalog matches “${query}”.` : 'Loading the catalog…'}
            </p>
          )}
          {results.map((r, i) => {
            const head = i === 0 || results[i - 1].kind !== r.kind ? GROUP[r.kind] : null;
            return (
              <div key={`${r.kind}:${r.id}`} role="presentation">
                {head && (
                  <div className={styles.group} role="presentation">
                    {query ? head : 'Labs'}
                  </div>
                )}
                <div
                  id={optId(i)}
                  role="option"
                  aria-selected={r === current}
                  className={styles.option}
                  onPointerMove={() => setActive(i)}
                  onClick={() => go(r)}
                >
                  <span className={styles.kind} aria-hidden="true">
                    <Icon
                      name={
                        r.kind === 'lab' ? 'layers' : r.kind === 'method' ? 'arrowRight' : 'target'
                      }
                      size={14}
                    />
                  </span>
                  <span className={styles.main}>
                    <span className={styles.title}>
                      {r.title}
                      {r.status === 'open' && r.kind === 'lab' && (
                        <span className={styles.open}>open</span>
                      )}
                    </span>
                    <span className={styles.sub}>{r.subtitle}</span>
                  </span>
                  {r.tex ? (
                    <span className={styles.tex}>
                      <Formula tex={r.tex} />
                    </span>
                  ) : r.meta ? (
                    <span className={styles.meta}>{r.meta}</span>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
        <div className={styles.foot} aria-hidden="true">
          <span>
            <Kbd>↑</Kbd>
            <Kbd>↓</Kbd> move
          </span>
          <span>
            <Kbd>↵</Kbd> open in its lab
          </span>
          <span className={styles.footNote}>
            {index ? `${index.methods.length} methods · ${index.problems.length} problems` : ''}
          </span>
        </div>
      </div>
    </dialog>
  );
}

/** "12 results: 2 labs, 9 methods, 1 problem". */
function describeCounts(results: readonly SearchItem[]): string {
  if (results.length === 0) return 'No results';
  const n = (kind: string) => results.filter((r) => r.kind === kind).length;
  const part = (k: number, one: string, many: string) =>
    k ? `${k} ${k === 1 ? one : many}` : null;
  const parts = [
    part(n('lab'), 'lab', 'labs'),
    part(n('method'), 'method', 'methods'),
    part(n('problem'), 'problem', 'problems'),
    part(n('research'), 'study', 'studies'),
  ].filter(Boolean);
  return `${results.length} ${results.length === 1 ? 'result' : 'results'}: ${parts.join(', ')}`;
}
