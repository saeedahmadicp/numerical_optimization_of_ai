import { Fragment, type ReactNode } from 'react';
import { hasSci, int, sci, sig, splitSci, vec } from '../../core/format';
import styles from './Num.module.css';

export interface NumProps {
  value: number | readonly number[] | null | undefined;
  /** 'sig' (default): significant digits, scientific below 10⁻³ and above 10⁵; 'sci'; 'int' (1,000). */
  mode?: 'sig' | 'sci' | 'int';
  digits?: number;
  className?: string;
}

/**
 * A typeset number: tabular lining figures, U+2212 minus, 1.3×10⁻¹¹ (with a real superscript),
 * `—` for undefined, `∞`. Digit grouping (1,000) only for counts (`mode="int"`), never for iterates.
 */
export function Num({ value, mode = 'sig', digits = 4, className }: NumProps) {
  let text: string;
  if (Array.isArray(value)) text = vec(value as number[], digits);
  else if (mode === 'int' && typeof value === 'number') text = int(value);
  else if (mode === 'sci') text = sci(value as number | null | undefined, digits);
  else text = sig(value as number | null | undefined, digits);
  return (
    <span className={className} style={{ fontVariantNumeric: 'tabular-nums lining-nums' }}>
      <SciText text={text} />
    </span>
  );
}

/**
 * Text with every `m×10ⁿ` typeset as a SciNum (Inter, tabular, a <sup> exponent); the rest of the
 * text is left as it is (so it keeps the face of its container, e.g. a mono table cell).
 */
export function SciText({ text }: { text: string }): ReactNode {
  if (!hasSci(text)) return text;
  return (
    <Fragment>
      {splitSci(text).map((part, i) =>
        typeof part === 'string' ? (
          part
        ) : (
          <SciParts key={i} mantissa={part.mantissa} exponent={part.exponent} />
        ),
      )}
    </Fragment>
  );
}

/** One number in scientific notation: `sci(value, digits)` typeset with a real superscript. */
export function SciNum({
  value,
  digits = 3,
  className,
}: {
  value: number | null | undefined;
  digits?: number;
  className?: string;
}) {
  return (
    <span className={className}>
      <SciText text={sci(value, digits)} />
    </span>
  );
}

function SciParts({ mantissa, exponent }: { mantissa: string; exponent: string }) {
  return (
    <span className={styles.sci}>
      {mantissa ? `${mantissa}×` : ''}10<sup>{exponent}</sup>
    </span>
  );
}
