/**
 * The key under the parameter plane: what each mark of the focused method's step means, drawn
 * with the same shapes as the canvas (models.ts `stepGeometry`). Identity is never color alone:
 * the method's name leads the row. Each entry's title gives the full sentence.
 */
import type { ReactNode } from 'react';
import { seriesVar } from '../../ui/colors';
import { Formula } from '../../ui/components/Formula';
import type { MethodKind } from './models';
import styles from './LeastSquaresLab.module.css';

type Glyph = 'ring' | 'dashedEllipse' | 'trials' | 'disk' | 'curve' | 'ellipse' | 'rejected';

function Svg({ g, slot }: { g: Glyph; slot: number }) {
  const c = seriesVar(slot);
  const line = { fill: 'none', stroke: c, strokeWidth: 1.5, strokeLinecap: 'round' as const };
  let body: ReactNode;
  switch (g) {
    case 'ring':
      body = <circle cx="8" cy="8" r="4" {...line} />;
      break;
    case 'dashedEllipse':
      body = (
        <ellipse
          cx="8"
          cy="8"
          rx="6.5"
          ry="3.8"
          transform="rotate(-25 8 8)"
          {...line}
          strokeWidth={1.2}
          strokeDasharray="2.5 2"
        />
      );
      break;
    case 'trials':
      body = (
        <>
          <circle cx="4" cy="8" r="2.6" {...line} strokeWidth={1.3} />
          <circle cx="11.5" cy="8" r="3" fill={c} />
        </>
      );
      break;
    case 'disk':
      body = (
        <circle cx="8" cy="8" r="5.5" {...line} strokeWidth={1.1} fill={c} fillOpacity={0.16} />
      );
      break;
    case 'curve':
      body = <path d="M2 13c2-6 6-9 12-10" {...line} strokeWidth={1.2} strokeDasharray="2.5 2" />;
      break;
    case 'ellipse':
      body = <ellipse cx="8" cy="8" rx="6.5" ry="3.8" transform="rotate(-25 8 8)" {...line} />;
      break;
    case 'rejected':
      body = (
        <>
          <path d="M2 12L9.6 6.2" {...line} strokeDasharray="2.5 2" />
          <circle cx="11.6" cy="4.8" r="2.6" {...line} strokeWidth={1.3} />
        </>
      );
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
  /** The full sentence (title and accessible description). */
  full: string;
}
const T = (tex: string) => <Formula tex={tex} />;

const KEYS: Record<MethodKind, Item[]> = {
  gauss_newton: [
    {
      g: 'ring',
      label: <>{T('\\mathbf{x}_k + \\mathbf{p}_k')} full step</>,
      full: 'Ring: the minimizer of the linear model L(h) = ½‖rₖ + Jₖh‖², the full Gauss–Newton step.',
    },
    {
      g: 'dashedEllipse',
      label: <>model level set {T('L(\\mathbf{h}) = L(\\mathbf{0})')}</>,
      full: 'Dashed ellipse: the model level set through xₖ; its axes come from JₖᵀJₖ.',
    },
    {
      g: 'trials',
      label: <>failed and accepted {T('\\alpha')}</>,
      full: 'Hollow dots: Armijo trials that failed. Solid dot: the accepted xₖ + αₖpₖ.',
    },
  ],
  levenberg_marquardt: [
    {
      g: 'disk',
      label: <>trust region {T('\\|\\mathbf{h}\\| \\le \\|\\mathbf{h}_k\\|')}</>,
      full: 'Disk: the trust region that μₖ enforces; μₖ is its Lagrange multiplier.',
    },
    {
      g: 'curve',
      label: (
        <>
          {T('\\mathbf{h}(\\mu)')}, {T('\\mu \\ge 0')}
        </>
      ),
      full: 'Dashed curve: every damped step, from Gauss–Newton at μ = 0 into xₖ along −∇f.',
    },
    {
      g: 'ellipse',
      label: <>model level set at {T('\\mathbf{h}_k')}</>,
      full: 'Ellipse: the model level set that touches the disk at xₖ + hₖ.',
    },
    {
      g: 'rejected',
      label: <>rejected, {T('\\varrho_k \\le 0')}</>,
      full: 'Dashed arrow, hollow end: the trial is rejected and μ grows by ν.',
    },
  ],
};

export function StageKey({ kind, name, slot }: { kind: MethodKind; name: string; slot: number }) {
  return (
    <div className={styles.key} role="note" aria-label={`Key for ${name}`}>
      <span className={styles.keyName}>
        <span className={styles.keySwatch} style={{ background: seriesVar(slot) }} />
        {name}
      </span>
      {KEYS[kind].map((it) => (
        <span key={it.g} className={styles.keyItem} title={it.full}>
          <Svg g={it.g} slot={slot} />
          {it.label}
          <span className="visually-hidden">: {it.full}</span>
        </span>
      ))}
    </div>
  );
}
