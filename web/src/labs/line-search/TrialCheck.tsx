/**
 * "This trial": the acceptance tests of the focused method at the current trial, with the
 * numbers of that trial filled into each inequality, the verdict of each test as the method
 * recorded it (Step.info.conditions), and what the search did next (read from the next step).
 */
import type { Result, Step } from '../../core/types';
import type { LineSearchKind } from '../../methods/line_search/methods';
import { Formula, Swatch } from '../../ui/components';
import { EXACT_C2, trialOf, zoomInterpolant, type Trial } from './geometry';
import { fmtNum, interpWords, texNum } from './format';
import styles from './LineSearchLab.module.css';

export interface TrialCheckProps {
  name: string;
  slot: number;
  kind: LineSearchKind;
  trace: readonly Step[];
  k: number;
  result: Result;
}

interface Row {
  id: string;
  name: string;
  tex: string;
  verdict: boolean | null;
}

const PHI_DIFF = '\\varphi(\\alpha_k) - \\varphi(0)';

function rows(kind: LineSearchKind, t: Trial): Row[] {
  const { alpha: a, phi, dphi, phi0, dphi0, c1 } = t;
  const c2 = t.c2 ?? 0;
  const cond = t.conditions;
  const diff = phi - phi0;
  const rel = (ok: boolean | null, yes: string, no: string) => (ok === false ? no : yes);
  /**
   * `value ≤ bound` with each side's symbols set under its number. The label names the quantity
   * the test checks, so it is set one step above script style (about 0.8 em, its own subscripts
   * near 9.4 px) instead of the default script size, whose subscripts fall to 8.4 px.
   */
  const ub = (val: number | null, sym: string) =>
    `\\underbrace{${texNum(val)}}_{\\textstyle\\footnotesize ${sym}}`;
  const armijo = (label: string, c: string): Row => ({
    id: 'armijo',
    name: label,
    tex: `${ub(diff, PHI_DIFF)} ${rel(cond.armijo, '\\le', '>')} ${ub(c1 * a * dphi0, `${c}\\,\\alpha_k\\varphi'(0)`)}`,
    verdict: cond.armijo ?? null,
  });
  const notEvaluated = (id: string, name: string): Row => ({ id, name, tex: '', verdict: null });
  switch (kind) {
    case 'backtracking':
      return [armijo('Sufficient decrease (Armijo)', 'c_1')];
    case 'strong_wolfe':
      return [
        armijo('Sufficient decrease', 'c_1'),
        dphi === null
          ? notEvaluated('strong_curvature', 'Strong curvature')
          : {
              id: 'strong_curvature',
              name: 'Strong curvature',
              tex: `${ub(Math.abs(dphi), "|\\varphi'(\\alpha_k)|")} ${rel(cond.strong_curvature, '\\le', '>')} ${ub(
                -c2 * dphi0,
                "c_2|\\varphi'(0)|",
              )}`,
              verdict: cond.strong_curvature ?? null,
            },
      ];
    case 'weak_wolfe':
      return [
        armijo('Sufficient decrease', 'c_1'),
        dphi === null
          ? notEvaluated('curvature', 'Curvature')
          : {
              id: 'curvature',
              name: 'Curvature',
              tex: `${ub(dphi, "\\varphi'(\\alpha_k)")} ${rel(cond.curvature, '\\ge', '<')} ${ub(
                c2 * dphi0,
                "c_2\\varphi'(0)",
              )}`,
              verdict: cond.curvature ?? null,
            },
      ];
    case 'goldstein':
      return [
        armijo('Not too long (upper line)', 'c'),
        {
          id: 'goldstein_lower',
          name: 'Not too short (lower line)',
          tex: `${ub(diff, PHI_DIFF)} ${rel(cond.goldstein_lower, '\\ge', '<')} ${ub(
            (1 - c1) * a * dphi0,
            "(1-c)\\,\\alpha_k\\varphi'(0)",
          )}`,
          verdict: cond.goldstein_lower ?? null,
        },
      ];
    case 'exact_quadratic':
      return [
        {
          id: 'model',
          name: 'Model step',
          tex: `\\alpha_1 = -\\frac{\\varphi'(0)}{\\mathbf p^\\top\\nabla^2 f(\\mathbf x_0)\\,\\mathbf p} = -\\frac{${texNum(
            dphi0,
          )}}{${texNum(t.pHp)}} = ${texNum(a)}`,
          verdict: null,
        },
        {
          id: 'decrease',
          name: 'Decrease',
          tex: `${ub(diff, PHI_DIFF)} ${rel(cond.decrease, '\\le', '>')} 0`,
          verdict: cond.decrease ?? null,
        },
        ...(dphi === null
          ? []
          : [
              {
                id: 'strong_curvature',
                name: 'Gradient check',
                tex: `${ub(Math.abs(dphi), "|\\varphi'(\\alpha_k)|")} ${rel(cond.strong_curvature, '\\le', '>')} ${ub(
                  -EXACT_C2 * dphi0,
                  `${EXACT_C2}\\,|\\varphi'(0)|`,
                )}`,
                verdict: cond.strong_curvature ?? null,
              },
            ]),
      ];
  }
}

/** One sentence: what the search did after this trial (from the next recorded step). */
function nextMove(
  kind: LineSearchKind,
  t: Trial,
  next: Trial | null,
  trace: readonly Step[],
): string {
  if (!next) return '';
  const a = fmtNum(next.alpha);
  const k1 = `α${sub(next.k)}`;
  switch (kind) {
    case 'backtracking':
      return `Too long: the next trial is ${k1} = ρα${sub(t.k)} = ${a}.`;
    case 'weak_wolfe':
    case 'goldstein': {
      if (!next.interval) return `Next trial: ${k1} = ${a}.`;
      const [lo, hi] = next.interval;
      const bracket = Number.isFinite(hi) ? `[${fmtNum(lo)}, ${fmtNum(hi)}]` : `[${fmtNum(lo)}, ∞)`;
      const why = t.conditions.armijo === false ? 'Too long' : 'Too short';
      const rule = Number.isFinite(hi) ? '(ℓ + u)/2' : '2ℓ';
      return `${why}: the bracket [ℓ, u] is now ${bracket}, and ${k1} = ${rule} = ${a}.`;
    }
    case 'strong_wolfe': {
      if (next.phase === 'expand') return `The slope is still steep: α doubles, ${k1} = ${a}.`;
      const ip = zoomInterpolant(trace, next.k);
      const [lo, hi] = next.interval ?? [NaN, NaN];
      const how = interpWords(next.interp, ip?.kind);
      return `Zoom in [${fmtNum(lo)}, ${fmtNum(hi)}]: ${how} gives ${k1} = ${a}.`;
    }
    default:
      return '';
  }
}

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (k: number) => String(k).replace(/\d/g, (d) => SUBS[Number(d)]);

export function TrialCheck({ name, slot, kind, trace, k, result }: TrialCheckProps) {
  if (!trace.length) return null;
  const start = trialOf(trace[0]);
  const last = trace.length - 1;
  const kk = Math.max(0, Math.min(k, last));
  const head = (
    <div className={styles.checkHead}>
      <Swatch slot={slot} size={9} />
      <span className={styles.checkName}>{name}</span>
      <span className={styles.checkK}>
        {kk === 0 ? 'start' : `trial ${kk} of ${last}`}
        {kk > 0 && ` · ${trialOf(trace[kk]).phase}`}
      </span>
    </div>
  );

  if (kk === 0) {
    const first = trace[1] ? trialOf(trace[1]) : null;
    return (
      <div className={styles.check}>
        {head}
        <div className={styles.checkRow}>
          <span className={styles.checkLabel}>Slope along 𝐩</span>
          <Formula
            tex={`\\varphi'(0) = \\nabla f(\\mathbf x_0)^\\top\\mathbf p = ${texNum(start.dphi0)} ${
              start.dphi0 < 0 ? '< 0' : '\\ge 0'
            }`}
            fallback
          />
        </div>
        <p className={styles.checkNote}>
          {start.dphi0 < 0
            ? first
              ? `𝐩 is a descent direction. φ(0) = ${fmtNum(start.phi0)}; the first trial is α₁ = ${fmtNum(first.alpha)}.`
              : `𝐩 is a descent direction, but the search made no trial: ${result.message}.`
            : `No search: ${result.message}.`}
        </p>
      </div>
    );
  }

  const t = trialOf(trace[kk]);
  const next = kk < last ? trialOf(trace[kk + 1]) : null;
  const rs = rows(kind, t);
  let verdict: string;
  if (t.accepted) verdict = `Accepted: α = ${fmtNum(t.alpha)} is the step.`;
  else if (!next)
    verdict = `The search stopped here without an acceptable step: ${result.message}.`;
  else {
    verdict = nextMove(kind, t, next, trace);
    if (kind === 'strong_wolfe' && t.conditions.armijo === true && t.dphi === null)
      verdict = `φ(α${sub(t.k)}) is not below the best step so far. ${verdict}`;
  }
  return (
    <div className={styles.check}>
      {head}
      {rs.map((r) => (
        <div
          key={r.id}
          className={styles.checkRow}
          data-verdict={r.verdict === null ? 'none' : r.verdict ? 'pass' : 'fail'}
        >
          <span className={styles.checkLabel}>
            {r.name}
            {r.verdict !== null && (
              <span className={styles.checkMark} aria-label={r.verdict ? 'holds' : 'fails'}>
                {r.verdict ? ' ✓ holds' : ' ✗ fails'}
              </span>
            )}
          </span>
          {r.tex ? (
            <Formula tex={r.tex} display fallback fit />
          ) : (
            <p className={styles.checkSkip}>
              Not tested: the search evaluates φ′(α{sub(kk)}) only after the step passes its
              decrease tests.
            </p>
          )}
        </div>
      ))}
      <p className={styles.checkNote}>{verdict}</p>
    </div>
  );
}
