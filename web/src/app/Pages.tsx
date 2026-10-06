/** Page frame and the small pages: labs index, planned lab, 404, loading. */
import type { ReactNode } from 'react';
import { int } from '../core/format';
import { Button } from '../ui/components/Button';
import { Formula } from '../ui/components/Formula';
import { Icon } from '../ui/components/Icon';
import { useChartColors } from '../ui/theme';
import { useCanvas } from '../viz/useCanvas';
import type { LabEntry } from '../labs';
import { AppHeader, type Crumb } from './AppHeader';
import { REPO } from './catalog';
import { Footer } from './Footer';
import { LabsIndex } from './LabsIndex';
import { Lockup } from './Logo';
import { hashQuery, href, navigate, useHash } from './router';
import { drawThumbnail } from './thumbnails';
import { useCatalogIndex } from './useCatalogIndex';
import styles from './Pages.module.css';

/** Where a family lives in the Python package. */
const SOURCE: Record<string, string> = {
  systems: 'roots/systems.py',
  global: 'unconstrained/global_.py',
  least_squares: 'unconstrained/least_squares.py',
};
const sourcePath = (family: string) => `src/numopt/${SOURCE[family] ?? family}`;

/** Header, a 1240-px column, footer. */
export function PageFrame({
  title,
  lede,
  eyebrow,
  crumbs,
  children,
}: {
  title: ReactNode;
  lede?: ReactNode;
  eyebrow?: ReactNode;
  crumbs?: Crumb[];
  children?: ReactNode;
}) {
  return (
    <div className={styles.page}>
      <AppHeader crumbs={crumbs} />
      <main className={styles.container}>
        <header className={styles.head}>
          {/* A label line, not running text (its link stays undecorated). */}
          {eyebrow && <div className={styles.eyebrow}>{eyebrow}</div>}
          <h1 className={styles.title}>{title}</h1>
          {lede && <p className={styles.lede}>{lede}</p>}
        </header>
        {children}
      </main>
      <Footer />
    </div>
  );
}

export function LabsPage() {
  return (
    <PageFrame
      title="Labs"
      lede="Every family of the Python reference, grouped as a numerical-analysis syllabus would be. Open labs replay the methods iterate by iterate; planned labs list the methods they will compare."
    >
      <LabsIndex />
    </PageFrame>
  );
}

function BigThumb({ id }: { id: string }) {
  const colors = useChartColors();
  const { canvasRef } = useCanvas((ctx, s) => drawThumbnail(id, ctx, s.width, s.height, colors));
  return <canvas ref={canvasRef} aria-hidden="true" />;
}

/** A lab without `index.tsx`: what it will show, and the methods it will compare. */
export function PlannedLab({ lab }: { lab: LabEntry }) {
  const index = useCatalogIndex();
  const want = hashQuery(useHash()).get('m')?.split('~')[0] ?? null;
  const methods = index?.methods.filter((m) => (lab.families as string[]).includes(m.family)) ?? [];
  const problems = index?.problems.filter((p) => lab.problemKinds.includes(p.kind)) ?? [];
  return (
    <PageFrame
      crumbs={[{ label: 'Labs', href: href('/labs') }, { label: lab.title }]}
      eyebrow={
        <>
          <span className={styles.plannedDot} aria-hidden="true" />
          Planned lab · {lab.group}
        </>
      }
      title={lab.title}
      lede={lab.pitch}
    >
      <div className={styles.plannedGrid}>
        <div className={styles.plannedMain}>
          <div className={styles.problemBox}>
            <Formula tex={lab.problem} display fallback />
          </div>
          <p className={styles.note}>
            The lab is being built. The Python reference already has {int(lab.methodCount)}{' '}
            {lab.methodCount === 1 ? 'method' : 'methods'} for it; each will be replayed here
            iterate by iterate, beside its update rule and its source.
          </p>
          <h2 className={styles.h2}>Methods</h2>
          {!index && <p className={styles.note}>Loading the catalog…</p>}
          <ol className={styles.methodList}>
            {methods.map((m) => (
              <li key={m.id} className={styles.method} data-current={m.id === want || undefined}>
                <div className={styles.methodHead}>
                  <span className={styles.methodName}>{m.name}</span>
                  {m.order && <span className={styles.order}>{m.order}</span>}
                </div>
                <p className={styles.summary}>{m.summary}</p>
                {m.references[0] && <cite className={styles.cite}>{m.references.join('; ')}</cite>}
              </li>
            ))}
          </ol>
        </div>
        <aside className={styles.plannedSide}>
          <div className={styles.thumb}>
            <BigThumb id={lab.id} />
          </div>
          {problems.length > 0 && (
            <>
              <h2 className={styles.h3}>Test problems</h2>
              <ul className={styles.problemList}>
                {problems.map((p) => (
                  <li key={p.id}>
                    <span>{p.name}</span>
                    <code>{p.id}</code>
                  </li>
                ))}
              </ul>
            </>
          )}
          <a
            className={styles.sourceLink}
            href={`${REPO}/${SOURCE[lab.families[0]] ? 'blob' : 'tree'}/main/${sourcePath(lab.families[0])}`}
            target="_blank"
            rel="noreferrer"
          >
            <Icon name="code" size={14} /> The Python source <Icon name="external" size={12} />
          </a>
        </aside>
      </div>
    </PageFrame>
  );
}

export function NotFound() {
  return (
    <div className={styles.page}>
      <AppHeader />
      <main className={styles.center}>
        <div className={styles.box}>
          <Lockup size={28} />
          <h1 className={styles.titleSmall}>This page is not in the catalog.</h1>
          <p className={styles.lede}>The address does not match a lab, a method list or a page.</p>
          <div className={styles.row}>
            <Button variant="primary" icon="arrowLeft" onClick={() => navigate('/')}>
              Home
            </Button>
            <Button variant="secondary" onClick={() => navigate('/labs')}>
              All labs
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}

/**
 * The fallback while a lazy route loads. A lab page gets the lab header (full width, crumbs); a
 * site page (`site`) gets the same header its page will render (content width; the main nav, or
 * crumbs under `section`), so nothing in the header moves when the page arrives.
 */
export function PageLoading({
  title,
  section,
  site = false,
}: {
  title: string;
  section?: Crumb;
  site?: boolean;
}) {
  const parent = section ?? { label: 'Labs', href: href('/labs') };
  return (
    <div className={styles.page}>
      {site ? (
        <AppHeader crumbs={section ? [section, { label: title }] : undefined} />
      ) : (
        <AppHeader wide crumbs={[parent, { label: title }]} />
      )}
      <main className={styles.loading} aria-busy="true" aria-label={`Loading ${title}`}>
        <div className={styles.progress} />
      </main>
    </div>
  );
}

/**
 * The recoverable error panel a route shows when its page throws while rendering (see
 * `RouteBoundary`). "Try again" re-mounts the page with the same URL state; "Reset the lab"
 * drops the query (the state that most likely caused the error).
 */
export function PageError({
  title,
  section,
  message,
  onRetry,
}: {
  title: string;
  section?: Crumb;
  message: string;
  onRetry: () => void;
}) {
  const parent = section ?? { label: 'Labs', href: href('/labs') };
  const hash = window.location.hash;
  const hasQuery = hash.includes('?');
  return (
    <div className={styles.page}>
      <AppHeader wide crumbs={[parent, { label: title }]} />
      <main className={styles.center}>
        <div className={styles.box} role="alert">
          <h1 className={styles.titleSmall}>This page stopped with an error.</h1>
          <p className={styles.lede}>
            The rest of the site still works. Try again, or reset the page to its default state.
          </p>
          <code className={styles.errorMessage}>{message}</code>
          <div className={styles.row}>
            <Button variant="primary" onClick={onRetry}>
              Try again
            </Button>
            {hasQuery && (
              <Button
                variant="secondary"
                onClick={() => {
                  window.location.hash = hash.split('?')[0];
                  onRetry();
                }}
              >
                Reset to defaults
              </Button>
            )}
            <Button variant="secondary" icon="arrowLeft" onClick={() => navigate('/labs')}>
              All labs
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
