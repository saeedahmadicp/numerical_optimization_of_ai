import type { HTMLAttributes, ReactNode } from 'react';
import styles from './Misc.module.css';

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className={styles.kbd}>{children}</kbd>;
}

export type BadgeTone = 'neutral' | 'good' | 'warn' | 'bad' | 'accent';

/** A status pill. Status tones always carry a text label, never color alone. */
export function Badge({
  tone = 'neutral',
  dot,
  children,
}: {
  tone?: BadgeTone;
  dot?: boolean;
  children: ReactNode;
}) {
  return (
    <span className={`${styles.badge} ${tone !== 'neutral' ? styles[tone] : ''}`}>
      {dot && <span className={styles.dot} aria-hidden="true" />}
      {children}
    </span>
  );
}

export interface PanelProps extends Omit<HTMLAttributes<HTMLElement>, 'title'> {
  title?: ReactNode;
  actions?: ReactNode;
  flush?: boolean;
  as?: 'section' | 'div' | 'aside';
}

/** A bordered surface with an optional header row. */
export function Panel({
  title,
  actions,
  flush,
  as: Tag = 'section',
  children,
  className,
  ...rest
}: PanelProps) {
  return (
    <Tag className={`${styles.panel} ${className ?? ''}`} {...rest}>
      {(title || actions) && (
        <header className={styles.panelHeader}>
          {typeof title === 'string' ? <h2 className={styles.panelTitle}>{title}</h2> : title}
          {actions}
        </header>
      )}
      <div className={flush ? styles.flush : styles.panelBody}>{children}</div>
    </Tag>
  );
}

export const Card = Panel;

export function SectionLabel({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <div className={styles.sectionLabel} id={id}>
      {children}
    </div>
  );
}
