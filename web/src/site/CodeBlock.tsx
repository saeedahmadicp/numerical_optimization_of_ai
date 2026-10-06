/** A code sample that runs as shown: prompts, quiet highlighting, real output, a copy action. */
import { CopyButton } from '../ui/components/Copy';
import styles from './CodeBlock.module.css';

export interface CodeLine {
  /** `$`, `>>>`, or '' for a plain line. */
  prompt: string;
  code: string;
}

const KEYWORD = /^(import|from|for|in|def|return|as)$/;

function highlight(code: string) {
  const parts = code.split(/("[^"]*"|#.*$)/g);
  return parts.map((p, i) => {
    if (p.startsWith('"'))
      return (
        <span key={i} className={styles.str}>
          {p}
        </span>
      );
    if (p.startsWith('#'))
      return (
        <span key={i} className={styles.comment}>
          {p}
        </span>
      );
    return p
      .split(/(\bnumopt\b(?=\.)|\.\w+(?=\()|\b(?:import|from|for|in|def|return|as)\b|--[\w-]+)/g)
      .map((q, j) => {
        const k = `${i}-${j}`;
        if (/^\.\w+$/.test(q))
          return (
            <span key={k}>
              .<span className={styles.fn}>{q.slice(1)}</span>
            </span>
          );
        if (/^--/.test(q))
          return (
            <span key={k} className={styles.flag}>
              {q}
            </span>
          );
        if (KEYWORD.test(q))
          return (
            <span key={k} className={styles.kw}>
              {q}
            </span>
          );
        return q;
      });
  });
}

export function CodeBlock({
  lines,
  output,
  label,
}: {
  lines: CodeLine[];
  output?: string;
  /** Language name, shown in the corner and used for the copy button's name. */
  label: string;
}) {
  const text = lines.map((l) => l.code).join('\n');
  return (
    <figure className={styles.card}>
      <figcaption className={styles.head}>
        <span className={styles.lang}>{label}</span>
        <CopyButton text={text} label={`Copy the ${label} code`}>
          Copy
        </CopyButton>
      </figcaption>
      <pre className={styles.pre} tabIndex={0} aria-label={`${label} example`}>
        {lines.map((l, i) => (
          <span key={i} className={styles.line}>
            {l.prompt && (
              <span className={styles.prompt} aria-hidden="true">
                {l.prompt}{' '}
              </span>
            )}
            {highlight(l.code)}
            {'\n'}
          </span>
        ))}
        {output && <span className={styles.out}>{output}</span>}
      </pre>
    </figure>
  );
}
