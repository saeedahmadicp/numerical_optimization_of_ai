/**
 * #/research — the studies in research/: each one's question, its verified finding, whether it is
 * promoted into the package, and its lead figure. Everything is read from the READMEs at build
 * time (web/vite/research.ts); the HTML is the repository's own content with math typeset.
 */
import { int } from '../core/format';
import { Icon } from '../ui/components/Icon';
import { REPO } from '../app/catalog';
import type { StudySummary } from '../app/catalogTypes';
import { PageFrame } from '../app/Pages';
import { useResearch } from './useResearch';
import { labForFamily } from '../labs';
import styles from './Research.module.css';

const minutes = (words: number) => Math.max(1, Math.round(words / 220));

/**
 * The study's families by the names the Methods facets use ("Linear & integer programming"),
 * not by their ids ("lp"); a word that is not a family keeps its text, capitalized.
 */
function familyNames(cell: string): string {
  return cell
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t) => {
      const [id, ...rest] = t.split(/\s+(?=\()/);
      const lab = labForFamily(id.replace(/-/g, '_'));
      const name = lab?.title ?? id.charAt(0).toUpperCase() + id.slice(1).replace(/_/g, ' ');
      return [name, ...rest].join(' ');
    })
    .join(' · ');
}

function StudyCard({ s, n }: { s: StudySummary; n: number }) {
  return (
    <li className={styles.study}>
      <a
        className={styles.studyFigure}
        href={`#/research/${s.id}`}
        tabIndex={-1}
        aria-hidden="true"
      >
        {s.figure ? (
          <img src={s.figure} alt="" loading="lazy" decoding="async" />
        ) : (
          <span className={styles.noFigure}>No figure</span>
        )}
      </a>
      <div className={styles.studyBody}>
        <p className={styles.studyMeta}>
          <span className={styles.studyNum}>{String(n).padStart(2, '0')}</span>
          {s.family && <span className={styles.family}>{familyNames(s.family)}</span>}
          <span>
            {minutes(s.words)} min read · {int(s.figures)} {s.figures === 1 ? 'figure' : 'figures'}
          </span>
        </p>
        <h2 className={styles.studyTitle}>
          <a href={`#/research/${s.id}`}>{s.title}</a>
        </h2>
        {s.tags && <p className={styles.tags}>{s.tags}</p>}
        <dl className={styles.qa}>
          <div>
            <dt>Question</dt>
            <dd className="md-inline" dangerouslySetInnerHTML={{ __html: s.question }} />
          </div>
          {s.finding && (
            <div>
              <dt>Finding</dt>
              <dd className="md-inline" dangerouslySetInnerHTML={{ __html: s.finding }} />
            </div>
          )}
          {s.promote && (
            <div>
              <dt>Promoted</dt>
              <dd className="md-inline" dangerouslySetInnerHTML={{ __html: s.promote }} />
            </div>
          )}
        </dl>
        <a className={styles.read} href={`#/research/${s.id}`}>
          Read the study <Icon name="arrowRight" size={12} />
        </a>
      </div>
    </li>
  );
}

export default function ResearchPage() {
  const mod = useResearch();
  const studies = mod.studies;
  return (
    <PageFrame
      title="Research"
      lede="Where new methods, and improvements to existing ones, are tested before they enter the package. Each study asks one falsifiable question, runs a budget-matched experiment against numopt's baselines, reports where the new method loses, and regenerates every number from one command."
    >
      <div className={styles.indexBar}>
        <p className={styles.count}>
          {int(studies.length)} studies · <a href="#/research/about">How a study is run</a> ·{' '}
          <a href={`${REPO}/tree/main/research`} target="_blank" rel="noreferrer">
            research/ on GitHub <Icon name="external" size={11} />
          </a>
        </p>
      </div>
      <ol className={styles.studies}>
        {studies.map((s, i) => (
          <StudyCard key={s.id} s={s} n={i + 1} />
        ))}
      </ol>
    </PageFrame>
  );
}
