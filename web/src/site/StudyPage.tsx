/**
 * #/research/<id> — one study, rendered from its README at build time (math typeset with KaTeX,
 * figures served beside the app). A sticky table of contents follows the reading position;
 * `?s=<slug>` scrolls to a section, so every heading has a shareable link.
 */
import { useEffect, useRef, useState } from 'react';
import { usePrefersReducedMotion } from '../play/reducedMotion';
import { Icon } from '../ui/components/Icon';
import { REPO } from '../app/catalog';
import { NotFound, PageFrame } from '../app/Pages';
import { hashQuery, useHash } from '../app/router';
import { useStudyDoc } from './useResearch';
import styles from './Research.module.css';

const ABOUT = 'about';

/** The heading nearest above the reading line. */
function useActiveSection(
  root: React.RefObject<HTMLElement | null>,
  ready: boolean,
): string | null {
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    const el = root.current;
    if (!ready || !el) return;
    const heads = [...el.querySelectorAll<HTMLElement>('h2[id], h3[id]')];
    let raf = 0;
    const update = () => {
      raf = 0;
      const line = 140;
      let current: string | null = null;
      for (const h of heads) {
        if (h.getBoundingClientRect().top - line <= 0) current = h.id;
        else break;
      }
      setActive(current);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      cancelAnimationFrame(raf);
    };
  }, [root, ready]);
  return active;
}

export default function StudyPage({ id }: { id: string }) {
  const doc = useStudyDoc(id);
  const hash = useHash();
  const section = hashQuery(hash).get('s');
  const article = useRef<HTMLElement>(null);
  const reduced = usePrefersReducedMotion();
  const active = useActiveSection(article, !!doc);
  const [tocOpen, setTocOpen] = useState(false);

  // ?s=<slug> → scroll to that heading (on load and on every in-page link).
  useEffect(() => {
    if (!doc || !section) return;
    const target = document.getElementById(section);
    if (!target) return;
    const raf = requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
      target.setAttribute('tabindex', '-1');
      target.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(raf);
  }, [doc, section, reduced]);

  useEffect(() => {
    if (doc) document.title = `${doc.title} · Research · numopt`;
  }, [doc]);

  if (doc === null) return <NotFound />;
  const toc = doc.toc.filter((t) => t.depth === 2);
  const sub = (slug: string) => {
    const i = doc.toc.findIndex((t) => t.slug === slug);
    const out = [];
    for (let j = i + 1; j < doc.toc.length && doc.toc[j].depth > 2; j++) out.push(doc.toc[j]);
    return out;
  };
  const folder = id === ABOUT ? 'research' : `research/${id}`;

  return (
    <PageFrame
      crumbs={[
        { label: 'Research', href: '#/research' },
        { label: id === ABOUT ? 'How a study is run' : doc.title },
      ]}
      eyebrow={id === ABOUT ? 'The protocol' : 'Study'}
      title={id === ABOUT ? 'How a study is run' : doc.title}
      lede={
        id === ABOUT ? (
          'Every study follows one protocol: a falsifiable question, a method that follows the paper equation by equation, a fair budget-matched setup, performance and data profiles, an honest discussion, and one command that regenerates every number.'
        ) : (
          <span className="md-inline" dangerouslySetInnerHTML={{ __html: doc.lede }} />
        )
      }
    >
      {
        <>
          <div className={styles.studyBar}>
            <span>{Math.max(1, Math.round(doc.words / 220))} min read</span>
            <span>
              {doc.figures.length} {doc.figures.length === 1 ? 'figure' : 'figures'}
            </span>
            <a href={`${REPO}/tree/main/${folder}`} target="_blank" rel="noreferrer">
              <Icon name="github" size={13} /> {folder}/ <Icon name="external" size={11} />
            </a>
            {id !== ABOUT && <code className={styles.reproduce}>python {folder}/run.py</code>}
          </div>
          <div className={styles.studyLayout}>
            <nav className={styles.toc} aria-label="Contents" data-open={tocOpen || undefined}>
              <button
                type="button"
                className={styles.tocToggle}
                aria-expanded={tocOpen}
                onClick={() => setTocOpen((o) => !o)}
              >
                Contents <Icon name="chevronDown" size={12} />
              </button>
              <ol className={styles.tocList}>
                {toc.map((t) => {
                  const kids = sub(t.slug);
                  const on = active === t.slug || kids.some((k) => k.slug === active);
                  return (
                    <li key={t.slug}>
                      <a
                        href={`#/research/${id}?s=${t.slug}`}
                        aria-current={on ? 'location' : undefined}
                        onClick={() => setTocOpen(false)}
                      >
                        {t.text}
                      </a>
                      {on && kids.length > 0 && (
                        <ol>
                          {kids.map((k) => (
                            <li key={k.slug}>
                              <a
                                href={`#/research/${id}?s=${k.slug}`}
                                aria-current={active === k.slug ? 'location' : undefined}
                              >
                                {k.text}
                              </a>
                            </li>
                          ))}
                        </ol>
                      )}
                    </li>
                  );
                })}
              </ol>
            </nav>
            <article
              ref={article}
              className={`${styles.doc} md-doc`}
              dangerouslySetInnerHTML={{ __html: doc.html }}
            />
          </div>
        </>
      }
    </PageFrame>
  );
}
