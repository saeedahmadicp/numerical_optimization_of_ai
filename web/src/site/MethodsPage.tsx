/**
 * #/methods — every method of the Python reference, grouped by family in syllabus order, with
 * facets (family, derivatives used, rate, determinism) and a text filter. Facet counts follow the
 * other active filters. Every row opens the method's page (#/method/<id>). All state is in the URL.
 */
import { useMemo, useState } from 'react';
import { int } from '../core/format';
import { LABS } from '../labs';
import { Formula } from '../ui/components/Formula';
import { Icon } from '../ui/components/Icon';
import { MathText } from '../ui/components/MathText';
import { scriptsToMath } from '../ui/mathProse';
import { RateText } from '../ui/components/RateText';
import type { CatalogMethod } from '../app/catalogTypes';
import { CATALOG } from '../app/catalog';
import { PageFrame } from '../app/Pages';
import { useCatalogIndexNow } from '../app/useCatalogIndex';
import { codecs, useUrlState } from '../app/useUrlState';
import {
  matchesFilter,
  NEEDS_LABEL,
  NEEDS_TEX,
  needsClass,
  RATE_LABEL,
  RATE_ORDER,
  rateClass,
  type MethodFilter,
  type NeedsClass,
} from './methodMeta';
import styles from './Methods.module.css';

const NEEDS_ORDER: NeedsClass[] = ['f', 'grad', 'hess', 'other'];
const FAMILY_CODEC = codecs.string;

interface Facet {
  key: keyof Omit<MethodFilter, 'q'>;
  title: string;
  options: { value: string; label: string; tex?: string }[];
  value: string;
  set: (v: string) => void;
}

function FacetGroup({
  facet,
  methods,
  filter,
}: {
  facet: Facet;
  methods: readonly CatalogMethod[];
  filter: MethodFilter;
}) {
  // Count with every other facet applied (faceted search), so a count says what a click shows.
  const base = { ...filter, [facet.key]: '' };
  const counts = new Map<string, number>();
  for (const m of methods) {
    if (!matchesFilter(m, base)) continue;
    const v =
      facet.key === 'family'
        ? m.family
        : facet.key === 'needs'
          ? needsClass(m.needs)
          : facet.key === 'rate'
            ? rateClass(m.order)
            : m.deterministic
              ? 'deterministic'
              : 'stochastic';
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const id = `facet-${facet.key}`;
  return (
    <div className={styles.facet} role="group" aria-labelledby={id}>
      <h2 id={id} className={styles.facetTitle}>
        {facet.title}
        {facet.value && (
          <button type="button" className={styles.facetClear} onClick={() => facet.set('')}>
            Clear<span className="visually-hidden"> {facet.title}</span>
          </button>
        )}
      </h2>
      <ul className={styles.facetList}>
        {facet.options.map((o) => {
          const n = counts.get(o.value) ?? 0;
          const on = facet.value === o.value;
          return (
            <li key={o.value}>
              <button
                type="button"
                className={styles.facetOption}
                aria-pressed={on}
                disabled={!on && n === 0}
                onClick={() => facet.set(on ? '' : o.value)}
              >
                <span className={styles.facetLabel}>{o.label}</span>
                <span className={styles.facetCount}>{int(n)}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function NeedsChips({ needs }: { needs: readonly string[] }) {
  return (
    <span className={styles.needs} aria-label={`Needs: ${needs.join(', ')}`}>
      {needs.map((n) => (
        <span key={n} className={styles.need} aria-hidden="true">
          {NEEDS_TEX[n] ? <Formula tex={NEEDS_TEX[n]} fallback /> : n}
        </span>
      ))}
    </span>
  );
}

export default function MethodsPage() {
  const index = useCatalogIndexNow();
  const [q, setQ] = useUrlState('q', '');
  const [family, setFamily] = useUrlState('family', '', FAMILY_CODEC);
  const [needs, setNeeds] = useUrlState('needs', '');
  const [rate, setRate] = useUrlState('rate', '');
  const [det, setDet] = useUrlState('det', '');
  const [filtersOpen, setFiltersOpen] = useState(false);
  const filter: MethodFilter = { q, family, needs, rate, det };
  const methods = index.methods;

  const groups = useMemo(
    () =>
      LABS.map((lab) => ({
        lab,
        methods: methods
          .filter((m) => (lab.families as string[]).includes(m.family))
          .filter((m) => matchesFilter(m, { q, family, needs, rate, det }))
          .sort((a, b) => a.name.localeCompare(b.name)),
      })).filter((g) => g.methods.length > 0),
    [methods, q, family, needs, rate, det],
  );
  const shown = groups.reduce((n, g) => n + g.methods.length, 0);
  const active = [family, needs, rate, det].filter(Boolean).length;

  const facets: Facet[] = [
    {
      key: 'family',
      title: 'Family',
      value: family,
      set: setFamily,
      options: LABS.flatMap((l) => l.families.map((f) => ({ value: f, label: l.title }))),
    },
    {
      key: 'needs',
      title: 'Derivatives',
      value: needs,
      set: setNeeds,
      options: NEEDS_ORDER.map((n) => ({ value: n, label: NEEDS_LABEL[n] })),
    },
    {
      key: 'rate',
      title: 'Rate',
      value: rate,
      set: setRate,
      options: RATE_ORDER.map((r) => ({ value: r, label: RATE_LABEL[r] })),
    },
    {
      key: 'det',
      title: 'Randomness',
      value: det,
      set: setDet,
      options: [
        { value: 'deterministic', label: 'Deterministic' },
        { value: 'stochastic', label: 'Seeded (Mulberry32)' },
      ],
    },
  ];

  const clearAll = () => {
    setFamily('');
    setNeeds('');
    setRate('');
    setDet('');
    setQ('');
  };

  return (
    <PageFrame
      title="Methods"
      lede={`${int(CATALOG.methods)} methods in ${CATALOG.families} families. Each one names its source — book, algorithm and equation — states its stopping test, and is replayed in TypeScript against the Python record.`}
    >
      <div className={styles.layout}>
        <aside className={styles.facets} aria-label="Filters" data-open={filtersOpen || undefined}>
          <button
            type="button"
            className={styles.facetsToggle}
            aria-expanded={filtersOpen}
            onClick={() => setFiltersOpen((o) => !o)}
          >
            <Icon name="sliders" size={14} />
            Filters{active > 0 && ` · ${active}`}
            <Icon name="chevronDown" size={12} />
          </button>
          <div className={styles.facetsBody}>
            {facets.map((f) => (
              <FacetGroup key={f.key} facet={f} methods={methods} filter={filter} />
            ))}
          </div>
        </aside>

        <div className={styles.main}>
          <div className={styles.toolbar}>
            <label className={styles.filter}>
              <Icon name="search" size={15} />
              <span className="visually-hidden">Filter methods</span>
              <input
                type="search"
                value={q}
                placeholder="Name, id, rate, parameter or source…"
                onChange={(e) => setQ(e.target.value)}
                spellCheck={false}
              />
            </label>
            <span className={styles.count} aria-live="polite">
              {`${int(shown)} of ${int(methods.length)}`}
            </span>
            {(active > 0 || q) && (
              <button type="button" className={styles.clearAll} onClick={clearAll}>
                Clear all
              </button>
            )}
          </div>

          {groups.map(({ lab, methods: ms }) => (
            <section key={lab.id} className={styles.group} aria-labelledby={`m-${lab.id}`}>
              <header className={styles.groupHead}>
                <h2 id={`m-${lab.id}`} className={styles.groupTitle}>
                  {lab.title}
                  <span className={styles.groupCount}>{int(ms.length)}</span>
                </h2>
                <span className={styles.groupTex}>
                  <Formula tex={lab.problem} fallback />
                </span>
                <a className={styles.groupLab} href={`#/lab/${lab.id}`}>
                  Open the lab <Icon name="arrowRight" size={12} />
                </a>
              </header>
              <ul className={styles.rows}>
                {ms.map((m) => (
                  <li key={m.id}>
                    <a className={styles.row} href={`#/method/${m.id}`}>
                      <span className={styles.name}>
                        <span className={styles.nameText}>{m.name}</span>
                        <code className={styles.id}>{m.id}</code>
                      </span>
                      <span className={styles.summary}>
                        <MathText text={scriptsToMath(m.summary)} />
                      </span>
                      <span className={styles.facts}>
                        {m.order && (
                          <span className={styles.rate}>
                            <RateText text={scriptsToMath(m.order)} />
                          </span>
                        )}
                        <NeedsChips needs={m.needs} />
                        {!m.deterministic && <span className={styles.seeded}>seeded</span>}
                      </span>
                      <cite className={styles.cite}>{m.references[0] ?? ''}</cite>
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          ))}
          {shown === 0 && (
            <div className={styles.empty}>
              <p>No method matches these filters.</p>
              <button type="button" className={styles.clearAll} onClick={clearAll}>
                Clear all filters
              </button>
            </div>
          )}
        </div>
      </div>
    </PageFrame>
  );
}
