import { CopyButton } from '../ui/components/Copy';
import { REPO } from './catalog';
import styles from './CodeCard.module.css';

const INSTALL = `pip install git+${REPO}`;
const LINES: { prompt: string; code: string }[] = [
  { prompt: '$', code: INSTALL },
  { prompt: '>>>', code: 'import numopt' },
  { prompt: '>>>', code: 'from numopt import problems' },
  { prompt: '>>>', code: 'res = numopt.run("bfgs", problems.get("rosenbrock"), x0=[-1.2, 1.0])' },
  { prompt: '>>>', code: 'res.converged, res.n_iter' },
];
/** The real output of the snippet above (checked against the Python package). */
const OUTPUT = '(True, 38)';

function highlight(code: string) {
  // Quiet syntax color: the call and the package name in iris, strings in text-2.
  const parts = code.split(/("[^"]*")/g);
  return parts.map((p, i) =>
    p.startsWith('"') ? (
      <span key={i} className={styles.str}>
        {p}
      </span>
    ) : (
      p.split(/\b(run|import|from)\b/g).map((q, j) =>
        q === 'run' ? (
          <span key={`${i}-${j}`} className={styles.fn}>
            {q}
          </span>
        ) : q === 'import' || q === 'from' ? (
          <span key={`${i}-${j}`} className={styles.kw}>
            {q}
          </span>
        ) : (
          q
        ),
      )
    ),
  );
}

/** A snippet that runs as shown, with its real output. */
export function CodeCard() {
  const text = [INSTALL, ...LINES.slice(1).map((l) => l.code)].join('\n');
  return (
    <div className={styles.card}>
      <div className={styles.copy}>
        <CopyButton text={text} label="Copy the code">
          Copy
        </CopyButton>
      </div>
      <pre className={styles.pre} tabIndex={0} aria-label="Python example">
        {LINES.map((l, i) => (
          <span key={i} className={styles.line}>
            <span className={styles.prompt} aria-hidden="true">
              {l.prompt}{' '}
            </span>
            {highlight(l.code)}
            {'\n'}
          </span>
        ))}
        <span className={styles.out}>{OUTPUT}</span>
      </pre>
    </div>
  );
}
