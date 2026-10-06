/**
 * Home — calm and clean (web/README.md, "Home page — approved design"): a short hero with a live
 * showcase that cycles through a few labs' scenes, one quiet line of facts, then the labs gallery
 * grouped by topic, each card with its own visual that plays on hover or focus.
 */
import { lazy, Suspense } from 'react';
// Direct imports (not the components barrel): the home page stays small.
import { Button } from '../ui/components/Button';
import { int } from '../core/format';
import { LABS } from '../labs';
import { usePrefersReducedMotion } from '../play/reducedMotion';
import { preloadKatex } from '../ui/katex';
import { AppHeader } from './AppHeader';
import { CATALOG } from './catalog';
import { Footer } from './Footer';
import { HERO_SCENES } from './home/heroScenes';
import { HeroSkeleton } from './home/HeroSkeleton';
import { loadPreview } from './home/previews';
import { LabsIndex } from './LabsIndex';
import { navigate, parseRoute } from './router';
import styles from './Home.module.css';

const loadHero = () => import('./home/HeroShowcase');
const HeroShowcase = lazy(loadHero);

// When the app opens on the home page, fetch the hero's chunk, its first scene and KaTeX (for the
// math in the hero caption) at boot, side by side: none of them waits for the hero chunk to load
// and render first. Other routes do not pay for this; there the hero loads when the home page
// renders.
if (typeof window !== 'undefined' && parseRoute(window.location.hash).name === 'home') {
  void loadHero();
  void loadPreview(HERO_SCENES[0]);
  void preloadKatex();
}

export function Home() {
  const reduced = usePrefersReducedMotion();
  const toLabs = () => {
    const el = document.getElementById('labs-title');
    el?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
    el?.focus({ preventScroll: true });
  };
  return (
    <div className={styles.page}>
      <AppHeader />
      <main>
        <section className={`${styles.container} ${styles.hero}`} aria-labelledby="home-title">
          <div className={styles.heroText}>
            <h1 className={styles.title} id="home-title">
              <span>Numerical optimization,</span> <em>iterate by iterate.</em>
            </h1>
            <p className={styles.lede}>
              Pick a problem, race up to four methods across its landscape, and read every iterate
              beside the update rule that produced it.
            </p>
            <div className={styles.ctas}>
              <Button variant="primary" size="lg" iconRight="arrowRight" onClick={toLabs}>
                Open a lab
              </Button>
              <Button variant="secondary" size="lg" onClick={() => navigate('/methods')}>
                Browse methods
              </Button>
            </div>
            <p className={styles.facts}>
              {int(CATALOG.methods)} methods · {int(CATALOG.families)} families ·{' '}
              {int(CATALOG.problems)} problems · every TypeScript port replays the Python
              reference’s fixtures
            </p>
          </div>
          <div className={styles.heroFigure}>
            <Suspense fallback={<HeroSkeleton />}>
              <HeroShowcase />
            </Suspense>
          </div>
        </section>

        <section className={`${styles.container} ${styles.labs}`} aria-labelledby="labs-title">
          <div className={styles.sectionHead}>
            <h2 className={styles.sectionTitle} id="labs-title" tabIndex={-1}>
              Labs
            </h2>
            <p className={styles.sectionNote}>
              {int(LABS.length)} labs, grouped by topic. Hover or focus a card to watch it run.
            </p>
          </div>
          <LabsIndex headingLevel={3} />
        </section>
      </main>
      <Footer />
    </div>
  );
}
