/**
 * The labs gallery (home page and #/labs): one card per lab, grouped by topic in syllabus order.
 * Each card shows the lab's signature geometry from a real run and plays it on hover or focus
 * (src/app/home/LabCard.tsx). Labs and counts come from the lab registry (src/labs/index.ts).
 */
import { LAB_GROUPS, LABS } from '../labs';
import { LabCard } from './home/LabCard';
import styles from './LabsIndex.module.css';

export function LabsIndex({ headingLevel = 2 }: { headingLevel?: 2 | 3 }) {
  const GroupHeading = headingLevel === 2 ? 'h2' : 'h3';
  return (
    <div className={styles.index}>
      {LAB_GROUPS.map((group) => {
        const labs = LABS.filter((l) => l.group === group);
        if (!labs.length) return null;
        const gid = `group-${group.replace(/\W+/g, '-').toLowerCase()}`;
        return (
          <section key={group} className={styles.group} aria-labelledby={gid}>
            <GroupHeading id={gid} className={styles.groupTitle}>
              {group}
            </GroupHeading>
            <ul className={styles.grid}>
              {labs.map((lab) => (
                <li key={lab.id} className={styles.item}>
                  <LabCard lab={lab} headingLevel={headingLevel === 2 ? 3 : 4} />
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
