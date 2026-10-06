import { Icon } from '../ui/components/Icon';
import { REPO } from './catalog';
import { href } from './router';
import { Lockup } from './Logo';
import styles from './Footer.module.css';

/** Lockup, the tagline, project links. */
export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <Lockup size={18} />
          <p className={styles.tagline}>Numerical optimization, iterate by iterate.</p>
        </div>
        <nav className={styles.links} aria-label="Project">
          <a href={href('/python')}>
            <Icon name="code" size={14} /> Python package
          </a>
          <a href={`${REPO}/blob/main/docs/architecture.md`} target="_blank" rel="noreferrer">
            <Icon name="layers" size={14} /> Architecture
          </a>
          <a href={href('/research')}>
            <Icon name="book" size={14} /> Research
          </a>
          <a href={REPO} target="_blank" rel="noreferrer">
            <Icon name="github" size={14} /> GitHub
          </a>
          <a href={`${REPO}#citing`} target="_blank" rel="noreferrer">
            <Icon name="quote" size={14} /> Cite
          </a>
        </nav>
      </div>
    </footer>
  );
}
