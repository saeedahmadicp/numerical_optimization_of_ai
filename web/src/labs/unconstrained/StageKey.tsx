/**
 * The key under the landscape: what each mark of the focused method's step geometry means, drawn
 * with the same shapes as the canvas (geometry.ts). The method's name leads the row, so identity
 * is never color alone.
 */
import type { ReactNode } from 'react';
import { seriesVar } from '../../ui/colors';
import { Formula } from '../../ui/components/Formula';
import type { Kind } from './catalog';
import styles from './UnconstrainedLab.module.css';

type Glyph =
  | 'dot'
  | 'ring'
  | 'inkRing'
  | 'cross'
  | 'plus'
  | 'arrow'
  | 'dashArrow'
  | 'inkArrow'
  | 'inkDash'
  | 'ellipse'
  | 'inkEllipse'
  | 'disk'
  | 'dashDisk'
  | 'ray'
  | 'triangle'
  | 'dashTriangle'
  | 'stencil'
  | 'curve'
  | 'inkCurve'
  | 'polyline'
  | 'inkPolyline';

function Svg({ g, slot }: { g: Glyph; slot: number }) {
  const c = seriesVar(slot);
  const ink = 'var(--color-text)';
  const line = (stroke: string, dash?: string) => ({
    fill: 'none',
    stroke,
    strokeWidth: 1.5,
    strokeLinecap: 'round' as const,
    strokeDasharray: dash,
  });
  let body: ReactNode;
  switch (g) {
    case 'dot':
      body = <circle cx="8" cy="8" r="3" fill={c} />;
      break;
    case 'ring':
      body = <circle cx="8" cy="8" r="3" {...line(c)} />;
      break;
    case 'inkRing':
      body = <circle cx="8" cy="8" r="3" {...line(ink)} />;
      break;
    case 'cross':
      body = <path d="M3.5 8h9M8 3.5v9" {...line(c)} />;
      break;
    case 'plus':
      body = <path d="M3.5 8h9M8 3.5v9" {...line(ink)} />;
      break;
    case 'arrow':
    case 'dashArrow':
    case 'inkArrow':
    case 'inkDash': {
      const col = g === 'inkArrow' || g === 'inkDash' ? ink : c;
      const dash = g === 'dashArrow' || g === 'inkDash' ? '2.5 2' : undefined;
      body = (
        <>
          <path d="M2 12.5L10.5 5" {...line(col, dash)} />
          <path d="M13.5 2.5l-5.2 1 3.9 3.6z" fill={col} />
        </>
      );
      break;
    }
    case 'ellipse':
      body = (
        <ellipse
          cx="8"
          cy="8"
          rx="6.5"
          ry="3.6"
          transform="rotate(-28 8 8)"
          {...line(c)}
          fill={c}
          fillOpacity={0.14}
        />
      );
      break;
    case 'inkEllipse':
      body = (
        <ellipse
          cx="8"
          cy="8"
          rx="6.5"
          ry="3.6"
          transform="rotate(-28 8 8)"
          {...line(ink, '2.5 2')}
        />
      );
      break;
    case 'disk':
      body = <circle cx="8" cy="8" r="6" {...line(c)} fill={c} fillOpacity={0.12} />;
      break;
    case 'dashDisk':
      body = <circle cx="8" cy="8" r="6" {...line(c, '2.5 2')} />;
      break;
    case 'ray':
      body = (
        <>
          <path d="M1.5 12.5L14.5 3.5" {...line(c, '2.5 2')} />
          <circle cx="11" cy="6" r="1.8" {...line(c)} />
        </>
      );
      break;
    case 'triangle':
      body = (
        <path
          d="M2.5 13L8 3l5.5 10z"
          {...line(c)}
          fill={c}
          fillOpacity={0.14}
          strokeLinejoin="round"
        />
      );
      break;
    case 'dashTriangle':
      body = <path d="M2.5 13L8 3l5.5 10z" {...line(c, '2.5 2')} strokeLinejoin="round" />;
      break;
    case 'stencil':
      body = (
        <>
          <path d="M2 8h12M8 2v12" {...line(c)} />
          <circle cx="14" cy="8" r="1.5" {...line(c)} />
          <circle cx="2" cy="8" r="1.5" {...line(c)} />
        </>
      );
      break;
    case 'curve':
      body = <path d="M1.5 12c2.5-8 10.5-8 13 0" {...line(c)} />;
      break;
    case 'inkCurve':
      body = <path d="M1.5 12c2.5-8 10.5-8 13 0" {...line(ink)} />;
      break;
    case 'polyline':
      body = <path d="M2 13l5-4 7-6" {...line(c, '2.5 2')} />;
      break;
    case 'inkPolyline':
      body = <path d="M2 13c3-1 5-4 6-7s3-3 6-4" {...line(ink, '2.5 2')} />;
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
const TRIALS: Item = ['ray', <>search ray, rejected trials</>];

function itemsFor(kind: Kind, method: string, params: Record<string, unknown>): Item[] {
  switch (kind) {
    case 'line':
      return [
        TRIALS,
        ...(method === 'gradient_descent' && params.step_rule === 'exact_quadratic'
          ? ([
              ['inkCurve', <>level set {T('f = f(\\mathbf{x}_k)')} touched by the ray</>],
            ] as Item[])
          : []),
      ];
    case 'coord':
      return [['ray', <>coordinate axis {T('\\mathbf{x}_{k-1} + t\\,\\mathbf{e}_i')}</>]];
    case 'heavy':
      return [
        ['dashArrow', <>momentum {T('\\beta\\mathbf{v}_{k-1}')}</>],
        ['inkArrow', <>gradient step {T('-\\alpha\\nabla f(\\mathbf{x}_{k-1})')}</>],
      ];
    case 'nesterov':
      return [
        ['dashArrow', <>momentum {T('\\mu\\mathbf{v}_{k-1}')}</>],
        ['ring', <>look-ahead {T('\\mathbf{y}')}</>],
        ['inkArrow', <>{T('-\\alpha\\nabla f(\\mathbf{y})')}</>],
      ];
    case 'adaptive':
      return [
        [
          'ellipse',
          <>
            reachable steps{' '}
            {T('\\{-\\mathbf{D}\\mathbf{w} : \\|\\mathbf{w}\\| = \\|\\mathbf{u}\\|\\}')}
          </>,
        ],
        [
          'inkDash',
          <>
            direction {T('-\\mathbf{u}')} without {T('\\mathbf{D}')}
          </>,
        ],
      ];
    case 'newton':
      return [
        ['curve', <>model level set through {T('\\mathbf{x}_{k-1}')}</>],
        ['cross', <>its center, the Newton point {T('\\mathbf{x}^{N}')}</>],
        ...(method === 'pure_newton' ? [] : [TRIALS]),
      ];
    case 'qn':
      return [
        [
          'ellipse',
          <>
            model{' '}
            {T(
              'm(\\mathbf{d}) = \\nabla f^{\\top}\\mathbf{d} + \\frac12\\mathbf{d}^{\\top}H^{-1}\\mathbf{d}',
            )}{' '}
            through {T('\\mathbf{x}_{k-1}')}
          </>,
        ],
        ['inkEllipse', <>the same with {T('\\nabla^2 f')}</>],
        TRIALS,
      ];
    case 'cg':
      return [
        ['inkDash', T('-\\alpha\\nabla f')],
        ['dashArrow', T('\\alpha\\beta_{k-1}\\mathbf{d}_{k-2}')],
        ['inkCurve', <>level set where the last search ended</>],
        TRIALS,
      ];
    case 'tr': {
      const extra: Item[] =
        method === 'trust_region_dogleg'
          ? [['polyline', <>dogleg {T('\\mathbf{0} \\to \\mathbf{p}^{U} \\to \\mathbf{p}^{B}')}</>]]
          : method === 'trust_region_steihaug'
            ? [['polyline', <>inner CG iterates</>]]
            : [];
      return [
        ['disk', <>trust region {T('\\|\\mathbf{p}\\| \\le \\Delta_k')}</>],
        ['curve', <>model level set at the step</>],
        ['inkRing', <>Cauchy point {T('\\mathbf{p}^{C}')}</>],
        ['cross', <>Newton point {T('\\mathbf{p}^{B}')}</>],
        ...extra,
        ['dashDisk', <>next {T('\\Delta')}</>],
      ];
    }
    case 'nm':
      return [
        ['triangle', 'simplex'],
        ['dashTriangle', 'previous simplex'],
        [
          'ring',
          <>trial points {T('\\mathbf{x}_r, \\mathbf{x}_e, \\mathbf{x}_{oc}, \\mathbf{x}_{ic}')}</>,
        ],
      ];
    case 'powell':
      return [
        ['arrow', <>line minimizations along {T('\\mathbf{u}_1, \\dots, \\mathbf{u}_n')}</>],
        ['ring', <>extrapolated {T('\\mathbf{x}_E')}</>],
      ];
    case 'hj':
      return [
        ['stencil', <>exploratory probes {T('\\pm h\\,\\mathbf{e}_i')}</>],
        ['dashArrow', <>pattern move to {T('\\mathbf{p}')}</>],
      ];
    case 'compass':
      return [
        ['stencil', <>poll {T('\\mathbf{x} \\pm \\Delta\\,\\mathbf{e}_i')}</>],
        ['ring', 'evaluated'],
      ];
    case 'schedule':
      return [
        ['arrow', <>step {T('-\\tfrac{h}{L}\\nabla f')}</>],
        ['inkRing', <>where the step {T('1/L')} ends</>],
        ['dot', 'certified checkpoint'],
      ];
    case 'ogm':
      return [
        [
          'inkArrow',
          <>
            gradient step {T('-\\nabla f/L')} to {T('\\mathbf{y}_k')}
          </>,
        ],
        ['dashArrow', 'momentum'],
        ['dot', <>certified at {T('k = N')}</>],
      ];
    case 'fista':
      return [
        ['dashArrow', <>momentum to {T('\\mathbf{y}_k')}</>],
        ['inkArrow', <>{T('-\\nabla f(\\mathbf{y}_k)/L_k')}</>],
        ['ring', 'rejected backtracking trials'],
        ['inkRing', 'restart (momentum reset)'],
      ];
    case 'anderson':
      return [
        ['ring', <>history, ring size {T('|c_i|')}</>],
        ['plus', <>mix {T('\\bar{\\mathbf{x}} = \\sum c_i \\mathbf{x}_i')}</>],
        ['inkDash', 'gradient correction'],
      ];
    case 'arc':
      return [
        ['disk', <>ball {T('\\|\\mathbf{s}\\| = \\lambda/\\sigma')}</>],
        ['curve', <>cubic model level set through {T('\\mathbf{x}_{k-1}')}</>],
        ['inkRing', <>Cauchy point {T('\\mathbf{s}^{C}')}</>],
        ['cross', <>Newton point {T('\\mathbf{x}^{N}')}</>],
      ];
    case 'regnewton':
      return [
        ['ellipse', <>model with {T('\\nabla^2 f + \\lambda I')}</>],
        [
          'inkPolyline',
          <>path {T('\\lambda \\mapsto -(\\nabla^2 f + \\lambda I)^{-1}\\nabla f')}</>,
        ],
        ['cross', <>Newton point {T('\\mathbf{x}^{N}')}</>],
        ['ring', 'rejected trials'],
      ];
  }
}

export function StageKey({
  kind,
  method,
  name,
  slot,
  params,
}: {
  kind: Kind;
  method: string;
  name: string;
  slot: number;
  params: Record<string, unknown>;
}) {
  const items = itemsFor(kind, method, params);
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
    </div>
  );
}
