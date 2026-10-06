/**
 * The hero figure's frame before its scenes load: the same card, stage, legend block, controls row
 * and caption block as `HeroShowcase`, with the same reserved heights, so the page does not jump
 * when the showcase arrives (owner rule: no layout shift). Used as the Suspense fallback on the
 * home page and by `HeroShowcase` while the previews build.
 */
import styles from './HeroShowcase.module.css';

export function HeroSkeleton() {
  return (
    <div className={styles.figure} aria-hidden="true">
      <div className={styles.card}>
        <div className={styles.stage} />
        <div className={styles.legends} />
      </div>
      <div className={styles.controls} />
      <div className={styles.captions} />
    </div>
  );
}
