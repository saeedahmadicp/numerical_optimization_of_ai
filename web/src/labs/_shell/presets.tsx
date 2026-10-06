/**
 * "Try this" presets: one click sets the problem, the methods (with parameters), the start point
 * and any lab-specific keys, by writing the lab's URL state. Because the state lives in the URL,
 * a preset is also a shareable link and Back/Forward-safe.
 *
 *   const PRESETS: LabPreset[] = [
 *     { id: 'zigzag', title: 'Why steepest descent zigzags',
 *       note: 'Exact line search on an ill-conditioned quadratic: every step is orthogonal to the last.',
 *       problem: 'ill_conditioned_quadratic',
 *       methods: [{ id: 'gradient_descent', slot: 0, params: { step_rule: 'exact' } }],
 *       start: [-4, 1] },
 *   ];
 *   <LabShell presets={PRESETS} … />          // rendered as a "Try this" rail section
 *
 * A preset that must also drop lab keys the viewer edited (moved cities `c`, a capacity `C`)
 * names them in `clearKeys`, or the lab passes them once: `<LabShell presetClearKeys={['c', 'C']}>`.
 */
import { useId, useState, type ReactNode } from 'react';
import { useHash } from '../../app/router';
import { Formula } from '../../ui/components/Formula';
import { Icon } from '../../ui/components/Icon';
import { applyPreset, presetActive, type LabPreset } from './labState';
import styles from './presets.module.css';

/** Presets shown before "More examples" (the rest open on demand). */
const SHOWN = 4;

/**
 * The "Try this" list, kept short so the Problem and Methods sections stay near the top of the
 * rail: titles only, the note under the active preset, and at most four presets before a
 * "More examples" disclosure (it opens by itself when the active preset is one of the rest).
 */
export function TryThis({
  presets,
  clearKeys,
}: {
  presets: readonly LabPreset[];
  /** Lab URL keys every preset removes (moved cities, an edited capacity), see `LabPreset`. */
  clearKeys?: readonly string[];
}) {
  const hash = useHash();
  const listId = useId();
  const activeIndex = presets.findIndex((p) => presetActive(p, hash, clearKeys));
  const [more, setMore] = useState(false);
  const showAll = more || activeIndex >= SHOWN;
  const shown = showAll ? presets : presets.slice(0, SHOWN);
  return (
    <>
      <ul className={styles.list} id={listId}>
        {shown.map((p, i) => {
          const active = i === activeIndex;
          return (
            <li key={p.id}>
              <button
                type="button"
                className={styles.preset}
                aria-pressed={active}
                onClick={() => applyPreset(p, clearKeys)}
              >
                <span className={styles.title}>{p.title}</span>
                {p.note && active && <span className={styles.note}>{noteContent(p.note)}</span>}
              </button>
            </li>
          );
        })}
      </ul>
      {presets.length > SHOWN && activeIndex < SHOWN && (
        <button
          type="button"
          className={styles.more}
          aria-expanded={showAll}
          aria-controls={listId}
          onClick={() => setMore((m) => !m)}
        >
          {showAll ? 'Fewer examples' : `More examples (${presets.length - SHOWN})`}
          <Icon name="chevronDown" size={12} />
        </button>
      )}
    </>
  );
}

/** A string note with `$…$` segments → text and inline KaTeX; other nodes unchanged. */
function noteContent(note: ReactNode): ReactNode {
  if (typeof note !== 'string' || !note.includes('$')) return note;
  return note
    .split(/\$([^$]+)\$/)
    .map((part, i) => (i % 2 ? <Formula key={i} tex={part} fallback /> : part));
}
