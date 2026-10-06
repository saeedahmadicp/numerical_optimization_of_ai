/**
 * In place of "This step" when a certified schedule or OGM could not run because the problem
 * states no Lipschitz constant L (or μ) and is not a quadratic: what the method needs, the
 * curvature of f on the plotted region, and a button that sets that value in the method's
 * parameters.
 */
import { Button } from '../../ui/components';
import { seriesVar } from '../../ui/colors';
import type { Problem2D } from '../../core/types';
import { MathText } from './MathText';
import { curvatureOnView, niceUp } from './constants';
import { tn } from './stepTex';
import styles from './UnconstrainedLab.module.css';

const PLAIN = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 3 });
const plain = (x: number) => PLAIN.format(x);

export function ConstantHint({
  problem,
  method,
  name,
  slot,
  onUse,
}: {
  problem: Problem2D;
  method: string;
  name: string;
  slot: number;
  onUse: (params: Record<string, number>) => void;
}) {
  const c = curvatureOnView(problem);
  const needsMu = method === 'silver_gd_strongly_convex';
  const L = c ? niceUp(c.lMax) : null;
  const mu = c && c.muMin > 0 ? c.muMin : null;
  let text =
    'This method needs a global Lipschitz constant $L$ of $\\nabla f$' +
    (needsMu ? ' and the strong-convexity constant $\\mu$' : '') +
    '. The value 0 means “auto”, which reads them from the Hessian of a quadratic problem; ' +
    `${problem.name} is not a quadratic and states no constant.`;
  if (c && L !== null)
    text +=
      ` On the plotted region, $\\lambda_{\\max}(\\nabla^2 f) \\le ${tn(c.lMax, 3)}$` +
      (needsMu ? ` and $\\lambda_{\\min}(\\nabla^2 f) \\ge ${tn(c.muMin, 3)}$` : '') +
      (needsMu
        ? '. These bound the curvature on that region only, so a certificate computed with them '
        : '. This bounds the curvature on that region only, so a certificate computed with it ') +
      'holds while the run stays there.';
  const blocked = needsMu && mu === null;
  if (blocked)
    text +=
      ' $\\nabla^2 f$ has a negative eigenvalue on this region: $f$ is not strongly convex there, ' +
      'so the strongly convex schedule does not apply. Choose a convex quadratic problem.';
  const params: Record<string, number> = {};
  if (L !== null) params.L = L;
  if (needsMu && mu !== null) params.mu = Number(mu.toPrecision(2));
  return (
    <section className={styles.stepEq} aria-label={`${name} needs a constant`}>
      <p className={styles.stepEqHead}>
        <span className={styles.stepEqTitle}>
          Needs a constant
          <span className={styles.stepEqName}>
            <span
              className={styles.keySwatch}
              style={{ background: seriesVar(slot) }}
              aria-hidden="true"
            />
            {name}
          </span>
        </span>
      </p>
      <p className={styles.stepEqNote}>
        <MathText text={text} />
      </p>
      {L !== null && !blocked && (
        <div className={styles.hintAction}>
          <Button size="sm" variant="secondary" onClick={() => onUse(params)}>
            Use L = {plain(L)}
            {params.mu !== undefined ? `, μ = ${plain(params.mu)}` : ''}
          </Button>
        </div>
      )}
    </section>
  );
}
