/**
 * The key under the data plane: what each mark of the focused method's geometry means, drawn
 * with the same shapes as the canvas (DataStage.tsx). The method's name leads the row, so
 * identity is never color alone; each entry's title and accessible text give the full sentence.
 * After the name, the live value of that geometry at the playhead (the band's half-width, λ, the
 * objective). The bar has a fixed height (RegressionLab.module.css .keyBar): a new focus, a toggle or a new
 * dataset changes its entries, never the height of the plot above it.
 */
import type { ReactNode } from 'react';
import { seriesVar } from '../../ui/colors';
import { Formula } from '../../ui/components/Formula';
import type { Kind } from './model';
import styles from './RegressionLab.module.css';

type Glyph =
  | 'stick'
  | 'square'
  | 'fan'
  | 'huberBand'
  | 'minimaxBand'
  | 'fade'
  | 'ring'
  | 'pair'
  | 'dashRing'
  | 'truth';

function Svg({ g, slot }: { g: Glyph; slot: number }) {
  const c = seriesVar(slot);
  const ink = 'var(--color-text)';
  const line = (stroke: string, dash?: string, width = 1.5) => ({
    fill: 'none',
    stroke,
    strokeWidth: width,
    strokeLinecap: 'round' as const,
    strokeDasharray: dash,
  });
  let body: ReactNode;
  switch (g) {
    case 'stick':
      body = (
        <>
          <path d="M8 4.5v9" {...line(c)} />
          <circle cx="8" cy="3.5" r="2.4" fill={ink} />
        </>
      );
      break;
    case 'square':
      body = (
        <rect
          x="2.5"
          y="2.5"
          width="11"
          height="11"
          {...line(c, undefined, 1)}
          fill={c}
          fillOpacity={0.14}
        />
      );
      break;
    case 'fan':
      body = (
        <>
          <path d="M1.5 12.5C6 10 10 6 14.5 3" {...line(c, '2 2', 1.1)} opacity={0.9} />
          <path d="M1.5 12.5C6 11 10 9 14.5 7.5" {...line(c, '2 2', 1.1)} opacity={0.6} />
          <path d="M1.5 12.5H14.5" {...line(c, '2 2', 1.1)} opacity={0.35} />
        </>
      );
      break;
    case 'huberBand':
    case 'minimaxBand': {
      const dash = g === 'minimaxBand' ? '3.5 2' : '1.5 2';
      body = (
        <>
          <rect x="1.5" y="4" width="13" height="8" fill={c} fillOpacity={0.12} />
          <path d="M1.5 4H14.5M1.5 12H14.5" {...line(c, dash, 1.1)} />
        </>
      );
      break;
    }
    case 'fade':
      body = (
        <>
          <circle cx="4.5" cy="8" r="3" fill={ink} />
          <circle
            cx="11.5"
            cy="8"
            r="3"
            fill={ink}
            fillOpacity={0.2}
            stroke="var(--color-text-3)"
            strokeWidth={0.75}
          />
        </>
      );
      break;
    case 'ring':
      body = (
        <>
          <circle cx="8" cy="8" r="2.6" fill={ink} />
          <circle cx="8" cy="8" r="5.6" {...line(c)} />
        </>
      );
      break;
    case 'pair':
      body = (
        <>
          <path d="M3.5 12.5L12.5 3.5" {...line(c, '0.5 2.4')} />
          <circle cx="3" cy="13" r="1.8" fill={ink} />
          <circle cx="13" cy="3" r="1.8" fill={ink} />
        </>
      );
      break;
    case 'dashRing':
      body = (
        <>
          <circle cx="8" cy="8" r="2.6" fill={ink} />
          <circle cx="8" cy="8" r="6.2" {...line(c, '2 1.8', 1.3)} />
        </>
      );
      break;
    case 'truth':
      body = <path d="M1.5 8H14.5" {...line('var(--color-text-3)', '0.5 3', 1.6)} />;
      break;
  }
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" className={styles.keyGlyph}>
      {body}
    </svg>
  );
}

interface Item {
  g: Glyph;
  label: ReactNode;
  /** The full sentence (title and accessible text). */
  full: string;
}
const T = (tex: string) => <Formula tex={tex} />;

const STICKS: Item = {
  g: 'stick',
  label: <>residual {T('r_i = y_i - \\hat y(x_i)')}</>,
  full: 'Sticks: the residual of each point from the fit of the method drawn in detail.',
};
const SQUARES: Item = {
  g: 'square',
  label: <>squares, total area {T('\\textstyle\\sum r_i^2')}</>,
  full: 'Squares: side |rᵢ|, so their total area is the least-squares objective; one outlier’s square can outweigh all the others.',
};

const KEYS: Record<Kind, Item[]> = {
  ols: [],
  poly: [],
  ridge: [
    {
      g: 'fan',
      label: <>fits for {T('\\lambda = 10^{-8}, \\dots, 10^{4}')}</>,
      full: 'Dashed fan: the ridge fit at every second decade of λ, from nearly least squares to the flat mean ȳ.',
    },
  ],
  huber: [
    {
      g: 'huberBand',
      label: <>band</>,
      full: 'Band: |r| ≤ δσ̂ around the fit; squared loss inside, absolute loss outside.',
    },
    {
      g: 'fade',
      label: <>weight {T('w_i')}</>,
      full: 'Opacity: the IRLS weight; points outside the band fade with wᵢ = δσ̂/|rᵢ|.',
    },
  ],
  lad: [
    {
      g: 'fade',
      label: <>weight {T('w_i \\propto 1/|r_i|')}</>,
      full: 'Opacity: the IRLS weight of each point relative to the median weight.',
    },
    {
      g: 'ring',
      label: <>on the line, {T('|r_i| \\le \\varepsilon')}</>,
      full: 'Rings: the points the line passes through.',
    },
  ],
  theil: [
    {
      g: 'pair',
      label: <>median-slope pair</>,
      full: 'Dotted segment and rings: the pair whose slope is the median of all pairwise slopes (two pairs for an even count).',
    },
  ],
  minimax: [
    {
      g: 'minimaxBand',
      label: <>band</>,
      full: 'Band: ŷ ± |hₖ|, the leveled error of the reference line.',
    },
    {
      g: 'ring',
      label: <>reference points</>,
      full: 'Rings: the three reference points; the signs of their residuals alternate.',
    },
    {
      g: 'dashRing',
      label: <>enters next</>,
      full: 'Dashed ring: the point of largest error, exchanged into the reference next.',
    },
  ],
};

export function StageKey({
  kind,
  name,
  slot,
  squares,
  fan,
  truth,
  readout,
}: {
  kind: Kind;
  name: string;
  slot: number;
  /** The residual squares are drawn (least-squares methods, "Squares" on). */
  squares: boolean;
  /** The ridge λ path is drawn (degree ≥ 1). */
  fan: boolean;
  /** The noise-free function is drawn. */
  truth: boolean;
  /** The live value of the method's geometry at the playhead (TeX: δσ̂, |h_k|, λ, Σ rᵢ², …). */
  readout?: string | null;
}) {
  const items = [
    STICKS,
    ...(squares ? [SQUARES] : []),
    ...KEYS[kind].filter((it) => fan || it.g !== 'fan'),
  ];
  return (
    <div className={styles.key} role="note" aria-label={`Key for ${name}`}>
      <span className={styles.keyName}>
        <span className={styles.keySwatch} style={{ background: seriesVar(slot) }} />
        {name}
      </span>
      {readout && (
        <span className={styles.keyReadout}>
          <Formula tex={readout} />
        </span>
      )}
      {items.map((it) => (
        <span key={it.g} className={styles.keyItem} title={it.full}>
          <Svg g={it.g} slot={slot} />
          {it.label}
          <span className="visually-hidden">: {it.full}</span>
        </span>
      ))}
      {truth && (
        <span
          className={styles.keyItem}
          title="Dotted curve: the noise-free function the data were drawn from."
        >
          <Svg g="truth" slot={slot} />
          noise-free {T('f')}
          <span className="visually-hidden">
            : the dotted curve, the function the data were drawn from
          </span>
        </span>
      )}
    </div>
  );
}
