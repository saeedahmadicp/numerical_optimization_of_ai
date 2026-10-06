/**
 * A registry rate ("superlinear (≈ 1.84 with inverse quadratic steps); never much worse than
 * bisection"): the leading rate word is the Newsreader italic badge (brand.md §2, "Rate badge"),
 * the qualification after it is Inter prose with inline math, because Newsreader is display only
 * and never body text beside math (brand.md §3). A rate without a leading rate word is all prose.
 */
import { MathText } from './MathText';
import styles from './RateText.module.css';

const HEAD =
  /^((?:r-)?(?:quadratic|cubic|superlinear|linear|sublinear|finite|direct|exact|heuristic|geometric)\b)/i;

export function RateText({ text }: { text: string }) {
  const m = HEAD.exec(text.trim());
  if (!m)
    return (
      <span className={styles.prose}>
        <MathText text={text} />
      </span>
    );
  const rest = text.trim().slice(m[1].length);
  return (
    <>
      <span className={styles.word}>{m[1]}</span>
      {rest && (
        <span className={styles.prose}>
          <MathText text={rest} />
        </span>
      )}
    </>
  );
}
