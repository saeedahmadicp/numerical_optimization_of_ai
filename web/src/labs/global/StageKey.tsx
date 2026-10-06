/**
 * The key under the landscape: what each glyph of the focused method means, drawn with the same
 * shapes as the canvas (draw.ts), plus what the line follows. Identity is never color alone:
 * the method's name leads the row.
 */
import type { ReactNode } from 'react';
import { seriesVar } from '../../ui/colors';
import { Formula } from '../../ui/components/Formula';
import styles from './GlobalLab.module.css';

type Glyph =
  | 'dot'
  | 'ring'
  | 'cross'
  | 'square'
  | 'arrow'
  | 'ellipse'
  | 'diamond'
  | 'best'
  | 'box'
  | 'line'
  | 'mutation';

function Svg({ g, slot }: { g: Glyph; slot: number }) {
  const c = seriesVar(slot);
  const common = { fill: 'none', stroke: c, strokeWidth: 1.5, strokeLinecap: 'round' as const };
  let body: ReactNode;
  switch (g) {
    case 'dot':
      body = <circle cx="8" cy="8" r="3.5" fill={c} />;
      break;
    case 'ring':
      body = <circle cx="8" cy="8" r="3.5" {...common} />;
      break;
    case 'cross':
      body = <path d="M4.5 4.5l7 7M4.5 11.5l7-7" {...common} />;
      break;
    case 'square':
      body = <rect x="5" y="5" width="6" height="6" {...common} />;
      break;
    case 'mutation': {
      // The donor difference (dashed, faint ink) and the scaled step to the mutant (ink).
      const ink = { ...common, stroke: 'var(--color-text-2)' };
      body = (
        <>
          <path d="M1.5 13.5L9 10" {...ink} stroke="var(--color-text-3)" strokeDasharray="2 2" />
          <path d="M3 8.5L11.5 4.8" {...ink} />
          <path d="M14 3.7l-4.2-.3 1.6 3.6z" fill="var(--color-text-2)" />
        </>
      );
      break;
    }
    case 'arrow':
      body = (
        <>
          <path d="M2.5 11.5L11 4.5" {...common} />
          <path d="M13.5 2.5l-5 .8 3.6 3.6z" fill={c} />
        </>
      );
      break;
    case 'ellipse':
      body = <ellipse cx="8" cy="8" rx="6.5" ry="3.8" transform="rotate(-25 8 8)" {...common} />;
      break;
    case 'diamond':
      body = <path d="M8 3.5L12.5 8 8 12.5 3.5 8z" {...common} />;
      break;
    case 'best':
      body = (
        <>
          <circle cx="8" cy="8" r="5" {...common} />
          <circle cx="8" cy="8" r="1.6" fill={c} />
        </>
      );
      break;
    case 'box':
      body = <rect x="3" y="3" width="10" height="10" {...common} strokeDasharray="2 2" />;
      break;
    case 'line':
      body = <path d="M1.5 11c3-6 6-6 13-6" {...common} />;
      break;
  }
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" className={styles.keyGlyph}>
      {body}
    </svg>
  );
}

type Item = [Glyph, ReactNode];
const T = (tex: string) => <Formula tex={tex} />;

const KEYS: Record<string, Item[]> = {
  simulated_annealing: [
    ['dot', <>walker {T('\\mathbf{x}_k')}</>],
    ['ellipse', <>proposal {T('1\\sigma_k, 2\\sigma_k')}</>],
    ['ring', <>accepted {T('\\mathbf{y}')}</>],
    ['cross', 'rejected'],
    ['line', 'the chain'],
  ],
  particle_swarm: [
    ['dot', <>particle {T('\\mathbf{x}_i')}</>],
    ['arrow', <>velocity {T('\\mathbf{v}_i')}</>],
    ['ring', <>personal best {T('\\mathbf{p}_i')}</>],
    ['line', <>path of {T('\\mathbf{g}')}</>],
  ],
  differential_evolution: [
    ['dot', 'member'],
    ['arrow', <>trial {T('\\mathbf{u}_i')}</>],
    ['cross', 'lost trial'],
    ['mutation', <>mutation {T('F\\,(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3})')}</>],
    ['diamond', <>mutant {T('\\mathbf{v}_i')}</>],
    ['box', 'crossover corners'],
  ],
  cma_es: [
    ['ellipse', <>{T('\\mathcal{N}(\\mathbf{m}, \\sigma^2 C)')}: 1σ, 2σ</>],
    ['dot', <>{T('\\mu')} selected</>],
    ['ring', 'discarded'],
    ['line', <>path of {T('\\mathbf{m}_k')}</>],
  ],
  basin_hopping: [
    ['box', 'hop box'],
    ['line', 'Nelder–Mead descent'],
    ['square', 'minima found'],
    ['cross', 'rejected minimum'],
  ],
};

/** The DE mutation term of the selected strategy. */
const BEST_MUTATION: Item = [
  'mutation',
  <>mutation {T('F\\,(\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2})')}</>,
];

export function StageKey({
  method,
  name,
  slot,
  strategy,
}: {
  method: string;
  name: string;
  slot: number;
  /** DE strategy (rand/1/bin or best/1/bin). */
  strategy?: string;
}) {
  const items = (KEYS[method] ?? []).map((it) =>
    it[0] === 'mutation' && strategy === 'best/1/bin' ? BEST_MUTATION : it,
  );
  return (
    <div className={styles.key} aria-label={`Key for ${name}`} role="note">
      <span className={styles.keyName}>
        <span className={styles.keySwatch} style={{ background: seriesVar(slot) }} />
        {name}
      </span>
      {items.map(([g, text], i) => (
        <span key={i} className={styles.keyItem}>
          <Svg g={g} slot={slot} />
          {text}
        </span>
      ))}
      <span className={styles.keyItem}>
        <Svg g="best" slot={slot} />
        best so far
      </span>
    </div>
  );
}
