/**
 * The MethodCard of the differentiation lab: the textbook page for the current step h_k. The
 * shared card labels its cells f(x_k), x_k, α_k, which mean nothing here (the "iterate" is the
 * estimate D(h_k)), so this card keeps the shared order (head · rule · this step · status ·
 * intuition · strengths/weaknesses · sources) with the quantities of a difference quotient: the
 * rule with this step's numbers, the digits that cancel in the numerator, the error estimates
 * and the selection of h⋆.
 */
import type { RegisteredMethod } from '../../core/registry';
import { defaults } from '../../core/registry';
import type { Params, Result, Step } from '../../core/types';
import { sci, sig, sigFixed } from '../../core/format';
import { CopyButton, Formula, Menu, MethodChip, type MenuItem } from '../../ui/components';
import { StepBlock, evidence, pythonCall } from '../_shell';
import { STENCILS, fsum, type StencilMethod } from '../../methods/differentiation/methods';
import { alignedDigits, digitsLost, texNum, zeroNumeratorKind } from './model';
import styles from './DiffCard.module.css';
// The shared method card's source list (Inter 13 px, text-2): one citation style in every lab.
import shared from '../_shell/blocks.module.css';

export interface DiffCardProps {
  method: RegisteredMethod;
  slot: number;
  step?: Step;
  /** The previous step (Richardson: its row is the previous row of the table). */
  prev?: Step;
  result?: Result;
  error?: string;
  /** The exact derivative the error is measured against (f′ or f″ at x0). */
  exact: number | null;
  /** The last level K of the sweep (the step block's label reads "Level k of K"). */
  levels: number;
  call: { problem: string; x0: number; params: Params };
}

const DIGITS = -Math.log10(Number.EPSILON); // 15.95 significant digits in a double

const offsetTex = (o: number) =>
  o === 0 ? 'f(x_0)' : `f(x_0 ${o < 0 ? '-' : '+'} ${Math.abs(o) === 1 ? '' : Math.abs(o)}h)`;

/** The textbook optimal steps (Python `_H_OPT` and `complex_step`), typeset. */
const H_OPT_TEX: Record<string, string> = {
  forward_difference: '2\\sqrt{\\varepsilon}\\,\\max(1, |x_0|)',
  backward_difference: '2\\sqrt{\\varepsilon}\\,\\max(1, |x_0|)',
  central_difference: '(3\\varepsilon)^{1/3}\\max(1, |x_0|)',
  five_point_stencil: '(45\\varepsilon/4)^{1/5}\\max(1, |x_0|)',
  second_derivative_central: '(48\\varepsilon)^{1/4}\\max(1, |x_0|)',
  complex_step: '\\sqrt{6\\varepsilon}\\,\\max(1, |x_0|)',
};

/** The truncation order, typeset (the rate badge; DOCS.order is the plain-text form). */
const ORDER_TEX: Record<string, string> = {
  forward_difference: 'O(h)',
  backward_difference: 'O(h)',
  central_difference: 'O(h^2)',
  five_point_stencil: 'O(h^4)',
  second_derivative_central: 'O(h^2)',
  richardson_extrapolation: 'O(h_k^{2k+2})',
  complex_step: 'O(h^2)',
};

/** The rule with this step's numbers filled in. */
function filledRule(id: string, step: Step, prev?: Step): string | null {
  const k = step.k;
  const h = step.info.h as number;
  const D = step.info.estimate as number | null;
  if (id in STENCILS) {
    const st = STENCILS[id as StencilMethod];
    const fs = (step.info.stencil as [number, number | null][]).map(([, v]) => v ?? NaN);
    const num = fsum(st.ints.map((c, i) => c * fs[i]));
    const den =
      st.q === 2
        ? `(${texNum(h, 4)})^2`
        : st.den === 1
          ? texNum(h, 4)
          : `${st.den} \\times ${texNum(h, 4)}`;
    const name = st.q === 2 ? 'D_2' : 'D';
    return `${name}(h_{${k}}) = \\frac{${texNum(num, 5)}}{${den}} = ${texNum(D, 10, true)}`;
  }
  if (id === 'richardson_extrapolation') {
    const row = step.info.row as (number | null)[];
    if (k === 0 || !prev || row.length < 2) return `D(${k}, 0) = ${texNum(row[0], 12)}`;
    const j = row.length - 1;
    const prevRow = prev.info.row as (number | null)[];
    const a = row[j - 1],
      b = prevRow[j - 1];
    // The correction's numerator itself: its two operands agree in most digits late in the
    // sweep, so rounding them separately would print 1 − 1.
    const diff = a !== null && b !== null && b !== undefined ? a - b : null;
    const den = j <= 6 ? String(4 ** j - 1) : `4^{${j}} - 1`;
    return `D(${k}, ${j}) = ${texNum(a, 12)} + \\frac{${texNum(diff, 3)}}{${den}} = ${texNum(row[j], 12, true)}`;
  }
  if (id === 'complex_step') {
    return `D(h_{${k}}) = \\frac{${texNum(step.info.imag as number, 6)}}{${texNum(h, 4)}} = ${texNum(D, 10, true)}`;
  }
  return null;
}

function Cancellation({ id, step }: { id: string; step: Step }) {
  if (id === 'complex_step') {
    const im = step.info.imag as number;
    return (
      <div className={styles.cancel}>
        <h3 className={styles.sub}>Digits that cancel</h3>
        <p className={styles.cancelNote}>
          None. <Formula tex="\operatorname{Im} f(x_0 + ih)" /> = {sci(im, 4)} is computed directly,
          not as a difference of two nearly equal values, so it keeps all {DIGITS.toFixed(0)} digits
          for every h.
        </p>
      </div>
    );
  }
  const offsets =
    id === 'richardson_extrapolation' ? [-1, 1] : (STENCILS[id as StencilMethod]?.offsets ?? []);
  const pts = step.info.stencil as [number, number | null][];
  const fs = pts.map(([, v]) => v ?? NaN);
  const weights = step.info.weights as number[];
  const { text, common } = alignedDigits(fs);
  const lost = digitsLost(fs, weights);
  const second = id === 'second_derivative_central';
  const D = second ? 'D_2' : 'D';
  return (
    <div className={styles.cancel}>
      <h3 className={styles.sub}>Digits that cancel</h3>
      <dl className={styles.digits}>
        {text.map((s, i) => (
          <div key={i} className={styles.digitRow}>
            <dt>
              <Formula tex={offsetTex(offsets[i] ?? 0)} />
            </dt>
            <dd>
              <span className={styles.dim}>{s.slice(0, common)}</span>
              <span className={styles.kept}>{s.slice(common)}</span>
            </dd>
          </div>
        ))}
      </dl>
      <p className={styles.cancelNote}>
        {!Number.isFinite(lost) ? (
          lost === Infinity ? (
            <ZeroNumerator fs={fs} second={second} D={D} />
          ) : (
            'A value of f is not finite at this step.'
          )
        ) : lost < 0.5 ? (
          'No significant digits cancel at this step: the error is all truncation.'
        ) : (
          `The numerator keeps about ${Math.max(0, DIGITS - lost).toFixed(1)} of ${DIGITS.toFixed(0)} significant digits; ${lost.toFixed(1)} cancel. Each halving of h cancels ${second ? 'about 0.6 more digits (h² shrinks 4×)' : 'about 0.3 more'}.`
        )}
      </p>
    </div>
  );
}

/**
 * Why the weighted sum Σ cᵢ fᵢ is exactly 0. Equal values cancel trivially; the second difference
 * also vanishes when its two first differences agree to the last bit (Sterbenz: both are exact).
 */
function ZeroNumerator({ fs, second, D }: { fs: number[]; second: boolean; D: string }) {
  const kind = zeroNumeratorKind(fs, second);
  if (kind === 'equal')
    return (
      <>
        Every digit cancels: the values are bitwise equal, so the numerator is exactly 0 and{' '}
        <Formula tex={`${D}(h) = 0`} />.
      </>
    );
  if (kind === 'balanced')
    return (
      <>
        The values differ, but the weighted sum <Formula tex="\textstyle\sum_i c_i f_i" /> is
        exactly 0: the differences <Formula tex="f(x_0 + h) - f(x_0)" /> and{' '}
        <Formula tex="f(x_0) - f(x_0 - h)" /> are equal to the last bit (both{' '}
        {sci(fs[2] - fs[1], 3)}), so <Formula tex="D_2(h) = 0" />.
      </>
    );
  return (
    <>
      The values differ, but the weighted sum <Formula tex="\textstyle\sum_i c_i f_i" /> rounds to
      exactly 0, so <Formula tex={`${D}(h) = 0`} />.
    </>
  );
}

/** A tolerance as "10⁻⁶" (or "5×10⁻⁷"). */
const tolText = (v: number) => sci(v, 2).replace(/^1×/, '');

const show = (v: unknown, digits = 4): string => {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number')
    return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e3 ? sci(v, digits) : sig(v, digits + 1);
  return String(v);
};

export function DiffCard({
  method,
  slot,
  step,
  prev,
  result,
  error,
  exact,
  levels,
  call,
}: DiffCardProps) {
  const { spec, doc } = method;
  const second = spec.id === 'second_derivative_central';
  const dTex = second ? 'D_2' : 'D';
  const fTex = second ? "f''(x_0)" : "f'(x_0)";
  const params = { ...defaults(spec), ...call.params };
  const tol = params.tol as number;
  const filled = step ? filledRule(spec.id, step, prev) : null;
  const extra = result?.extra ?? {};
  const kBest = extra.k_best as number | null | undefined;
  const bound = extra.bound_best as number | null | undefined;
  const actual = extra.error as number | null | undefined;
  const best = kBest !== null && kBest !== undefined ? result?.trace[kBest] : undefined;

  const live: { tex: string; value: string }[] = step
    ? [
        { tex: 'h_k', value: show(step.info.h) },
        { tex: `${dTex}(h_k)`, value: sigFixed(step.info.estimate as number | null, 10) },
        { tex: fTex, value: sigFixed(exact, 10) },
        { tex: `|${dTex} - ${fTex}|`, value: show(step.info.error, 2) },
        { tex: '\\hat e_{\\mathrm{trunc}}', value: show(step.info.err_est, 2) },
        { tex: '\\hat e_{\\mathrm{round}}', value: show(step.info.roundoff, 2) },
      ]
    : [];

  const menu: MenuItem[] = [
    {
      kind: 'copy',
      label: 'Copy Python call',
      icon: 'code',
      text: () =>
        pythonCall(spec.id, {
          problem: call.problem,
          x0: call.x0,
          params: call.params,
          specs: spec.params,
        }),
    },
    ...(doc?.rule
      ? [{ kind: 'copy' as const, label: 'Copy update rule (LaTeX)', text: doc.rule }]
      : []),
    ...(step
      ? [
          {
            kind: 'copy' as const,
            label: 'Copy this estimate',
            text: () => String(step.info.estimate),
          },
        ]
      : []),
    {
      kind: 'link',
      label: 'Find in the method catalog',
      icon: 'book',
      href: `#/methods?q=${encodeURIComponent(spec.id)}`,
    },
  ];

  let status: { tone: 'good' | 'warn' | 'bad'; icon: string; text: string } | null = null;
  if (error) status = { tone: 'bad', icon: '×', text: `Could not run: ${error}` };
  else if (result && best)
    status = result.converged
      ? {
          tone: 'good',
          icon: '✓',
          text: `Converged at h⋆ = ${sci(best.info.h as number, 3)} (level ${kBest}): error bound ${sci(bound ?? NaN, 2)} ≤ ${tolText(tol)}·max(1, |D|).`,
        }
      : {
          tone: 'warn',
          icon: '◷',
          text: `Not converged: the smallest error bound, ${sci(bound ?? NaN, 2)} at h = ${sci(best.info.h as number, 3)}, is not confirmed below ${tolText(tol)}·max(1, |D|).`,
        };
  else if (result) status = { tone: 'warn', icon: '△', text: evidence(result.message, params) };

  const hOpt = extra.h_opt as number | null | undefined;
  const hMin = extra.h_min_error as number | null | undefined;

  return (
    <div className={styles.card} tabIndex={0} role="region" aria-label={`${spec.name} details`}>
      <div className={styles.head}>
        <MethodChip name={spec.name} slot={slot} />
        <span className={styles.headEnd}>
          <span className={styles.rate} title={`Truncation order: ${doc?.order ?? spec.order}`}>
            {ORDER_TEX[spec.id] ? <Formula tex={ORDER_TEX[spec.id]} /> : (doc?.order ?? spec.order)}
          </span>
          <Menu label={`More actions for ${spec.name}`} items={menu} />
        </span>
      </div>
      {doc?.rule && (
        <div className={styles.rule}>
          <Formula tex={doc.rule} display fit />
        </div>
      )}
      {step && (
        <StepBlock
          k={step.k}
          label={`Level ${step.k} of ${levels}`}
          reset={`${spec.id}~${slot}~${result?.trace.length ?? 0}`}
        >
          {filled && (
            <div className={styles.filled}>
              <Formula tex={filled} display fit />
            </div>
          )}
          <dl className={styles.live}>
            {live.map((q) => (
              <div key={q.tex} className={styles.liveCell}>
                <dt className={styles.liveLabel}>
                  <Formula tex={q.tex} />
                </dt>
                <dd className={styles.liveValue}>{q.value}</dd>
              </div>
            ))}
          </dl>
          {!error && <Cancellation id={spec.id} step={step} />}
        </StepBlock>
      )}
      {status && (
        <div className={styles.statusBlock}>
          <p className={styles.status} data-tone={status.tone}>
            <span className={styles.statusIcon} aria-hidden="true">
              {status.icon}
            </span>
            {status.text}
          </p>
          {result && !error && (
            <ul className={styles.evidence}>
              {actual !== null && actual !== undefined && bound !== null && bound !== undefined && (
                <li>
                  Actual error <Formula tex={`|${dTex}(h^\\star) - ${fTex}|`} /> = {sci(actual, 2)}
                  {actual <= bound ? ' — within the bound.' : ' — the bound does not hold here.'}
                </li>
              )}
              {hMin !== null && hMin !== undefined && (
                <li>Smallest actual error of the sweep at h = {sci(hMin, 3)}.</li>
              )}
              {hOpt !== null &&
                hOpt !== undefined &&
                (spec.id === 'complex_step' ? (
                  <li>
                    No optimum: below <Formula tex="h_\varepsilon" /> ={' '}
                    <Formula tex={H_OPT_TEX.complex_step} /> = {sci(hOpt, 3)} the truncation{' '}
                    <Formula tex="h^2|f'''|/6" /> is under <Formula tex="\varepsilon|f'|" /> (when{' '}
                    <Formula tex="|f'''| \approx |f'|" />
                    ), and every smaller <Formula tex="h" /> is as good.
                  </li>
                ) : (
                  <li>
                    Textbook <Formula tex="h_{\mathrm{opt}}" /> ={' '}
                    <Formula tex={H_OPT_TEX[spec.id] ?? String(extra.h_opt_rule)} /> ={' '}
                    {sci(hOpt, 3)}.
                  </li>
                ))}
            </ul>
          )}
        </div>
      )}
      {(doc?.intuition || spec.summary) && (
        <p className={styles.intuition}>{doc?.intuition ?? spec.summary}</p>
      )}
      {(doc?.pros?.length || doc?.cons?.length) && (
        <div className={styles.proscons}>
          <div className={styles.pro}>
            <h3>Strengths</h3>
            <ul>
              {doc?.pros?.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
          <div className={styles.con}>
            <h3>Weaknesses</h3>
            <ul>
              {doc?.cons?.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {spec.references.length > 0 && (
        <div className={shared.refs}>
          <h3 className="visually-hidden">Sources</h3>
          {spec.references.map((r) => (
            <p key={r} className={shared.ref}>
              <cite>{r}</cite>
              <CopyButton text={r} label={`Copy the reference: ${r}`} className={shared.refCopy}>
                <span className="visually-hidden">Copy</span>
              </CopyButton>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
