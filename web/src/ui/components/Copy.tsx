import type { ReactNode } from 'react';
import { useCopied } from '../copy';
import { Icon, type IconName } from './Icon';
import styles from './Copy.module.css';

export interface CopyButtonProps {
  /** Text to copy (computed lazily when a function). */
  text: string | (() => string);
  children: ReactNode;
  icon?: IconName;
  className?: string;
  /** Accessible name when the visible label is not enough. */
  label?: string;
}

/** A quiet text button that copies and confirms inline ("Copied"). */
export function CopyButton({ text, children, icon = 'copy', className, label }: CopyButtonProps) {
  const [copied, copy] = useCopied();
  return (
    <button
      type="button"
      className={`${styles.copy} ${className ?? ''}`}
      aria-label={label}
      onClick={() => copy(typeof text === 'function' ? text() : text)}
    >
      <Icon name={copied === true ? 'check' : icon} size={14} />
      <span aria-live="polite">
        {copied === true ? 'Copied' : copied === 'failed' ? 'Copy blocked' : children}
      </span>
    </button>
  );
}
