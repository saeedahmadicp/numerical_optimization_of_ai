/**
 * The shared MethodCard (labs/_shell/blocks.tsx) in this lab's notation. The shared card names
 * the iterate 𝐱ₖ and the step αₖ; every other surface of this lab says 𝐰ₖ and ηₖ (the
 * parameters of a model, the learning rate). Two further changes: the update rule is set at
 * full size (the long rules are multi-line `aligned` blocks in methods.ts, not shrunk to fit)
 * and scrolls with a fade edge when it is still wider than the card.
 * TODO(platform): a `variable` prop on the shared MethodCard would replace this copy.
 */
import type { RegisteredMethod } from '../../core/registry';
import { defaults } from '../../core/registry';
import type { Params, Result, RunOptions, Step } from '../../core/types';
import { sci, sig, vec, int } from '../../core/format';
import { norm } from '../../core/linalg';
import {
  CopyButton,
  Formula,
  Menu,
  MethodChip,
  RateText,
  type MenuItem,
} from '../../ui/components';
import { describeResult, evidence, pythonCall } from '../_shell';
import { mathTextToPlain } from '../../ui/mathProse';
import { UPDATE_NOUN } from './geometry';
// The shared card's own stylesheet (read-only): this card must look exactly like it.
import styles from '../_shell/blocks.module.css';
import { ScrollFade } from './ScrollFade';

function readKey(step: Step, key: string): unknown {
  if (key.startsWith('info.')) return step.info[key.slice(5)];
  return (step as unknown as Record<string, unknown>)[key];
}

/** 4–6 significant digits; scientific below 10⁻³ and above 10⁵; vectors up to 3 entries. */
function show(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number')
    return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? sci(v, 3) : sig(v, 5);
  if (Array.isArray(v) && v.every((x) => typeof x === 'number'))
    return v.length <= 3 ? vec(v as number[], 4) : `‖·‖ = ${sig(norm(v as number[]), 4)}`;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  return String(v);
}

export interface StochasticCardProps {
  method: RegisteredMethod;
  slot: number;
  step?: Step;
  result?: Result;
  /** `LabRun.error`: the method rejected its input. Shown as an input error, not as divergence. */
  error?: string;
  /**
   * The run's problem, options and parameters: enables "Copy Python call" (the call that
   * reproduces this run in the reference package) and typesets tolerances in the status line.
   */
  call?: { problem?: string; options?: RunOptions; params?: Params };
}

/**
 * The lab's textbook page for the current step (portal-direction §3): head (chip · rate · menu),
 * the update rule between hairlines, this step's numbers, the status with its evidence, the
 * intuition, strengths and weaknesses, and the sources in italic with a copy action.
 */
export function StochasticCard({ method, slot, step, result, error, call }: StochasticCardProps) {
  const { spec, doc } = method;
  const order = doc?.order ?? spec.order;
  const vector = Array.isArray(step?.x);
  const xTex = vector ? '\\mathbf{w}_k' : 'w_k';
  const quantities = doc?.quantities ?? [];
  const live: { tex: string; value: string; wide?: boolean }[] = step
    ? [
        { tex: 'k', value: int(step.k) },
        { tex: `f(${xTex})`, value: show(step.fun) },
        ...(step.gradNorm !== null && step.gradNorm !== undefined
          ? [{ tex: `\\|\\nabla f(${xTex})\\|`, value: show(step.gradNorm) }]
          : []),
        { tex: xTex, value: show(step.x), wide: vector && (step.x as number[]).length > 1 },
        ...(step.stepSize !== null &&
        step.stepSize !== undefined &&
        !quantities.some((q) => q.key === 'stepSize')
          ? [{ tex: '\\eta_k', value: show(step.stepSize) }]
          : []),
        ...quantities.map((q) => ({ tex: q.tex, value: show(readKey(step, q.key)) })),
      ]
    : [];
  const status = result ? describeResult(result, error, UPDATE_NOUN) : null;
  const params = { ...defaults(spec), ...(call?.params ?? {}) };
  const why = result && !error && result.message ? evidence(result.message, params) : null;
  const menu: MenuItem[] = [
    ...(call
      ? [
          {
            kind: 'copy' as const,
            label: 'Copy Python call',
            icon: 'code' as const,
            text: () =>
              pythonCall(spec.id, {
                problem: call.problem,
                x0: call.options?.x0 as number[] | number | undefined,
                bracket: call.options?.bracket as number[] | undefined,
                seed: call.options?.seed as number | undefined,
                params: call.params,
                specs: spec.params,
              }),
          },
        ]
      : []),
    ...(doc?.rule
      ? [{ kind: 'copy' as const, label: 'Copy update rule (LaTeX)', text: doc.rule }]
      : []),
    ...(step
      ? [
          {
            kind: 'copy' as const,
            label: `Copy the current iterate`,
            text: () =>
              Array.isArray(step.x) ? `[${(step.x as number[]).join(', ')}]` : String(step.x),
          },
        ]
      : []),
    {
      kind: 'link' as const,
      label: 'Find in the method catalog',
      icon: 'book' as const,
      href: `#/methods?q=${encodeURIComponent(spec.id)}`,
    },
  ];
  return (
    // A keyboard-focusable scroll region (its content can be taller than the panel).
    <div className={styles.card} tabIndex={0} role="region" aria-label={`${spec.name} details`}>
      <div className={styles.cardHead}>
        <MethodChip name={spec.name} slot={slot} />
        <span className={styles.cardHeadEnd}>
          {order && (
            <span className={styles.rate} title={`Convergence rate: ${mathTextToPlain(order)}`}>
              <RateText text={order} />
            </span>
          )}
          <Menu label={`More actions for ${spec.name}`} items={menu} />
        </span>
      </div>
      {doc?.rule && (
        <ScrollFade className={styles.rule}>
          <Formula tex={doc.rule} display />
        </ScrollFade>
      )}
      {live.length > 0 && (
        <dl className={styles.live} aria-label="This step">
          {live.map((q) => (
            <div key={q.tex} className={styles.liveCell} data-wide={q.wide || undefined}>
              <dt className={styles.liveLabel}>
                <Formula tex={q.tex} />
              </dt>
              <dd className={styles.liveValue}>{q.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {status && (
        <div className={styles.statusBlock}>
          <p className={styles.status} data-tone={status.tone}>
            <span className={styles.statusIcon} aria-hidden="true">
              {status.icon}
            </span>
            {status.long}
          </p>
          {why && status.tone !== 'bad' && <p className={styles.evidence}>{why}</p>}
        </div>
      )}
      {(doc?.intuition || spec.summary) && (
        <p className={styles.intuition}>{doc?.intuition ?? spec.summary}</p>
      )}
      {(doc?.pros?.length || doc?.cons?.length) && (
        <div className={styles.proscons}>
          {doc?.pros?.length ? (
            <div className={styles.pro}>
              <h3>Strengths</h3>
              <ul>
                {doc.pros.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </div>
          ) : (
            <div />
          )}
          {doc?.cons?.length ? (
            <div className={styles.con}>
              <h3>Weaknesses</h3>
              <ul>
                {doc.cons.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      )}
      {spec.references.length > 0 && (
        <div className={styles.refs}>
          <h3 className="visually-hidden">Sources</h3>
          {spec.references.map((r) => (
            <p key={r} className={styles.ref}>
              <cite>{r}</cite>
              <CopyButton text={r} label={`Copy the reference: ${r}`} className={styles.refCopy}>
                <span className="visually-hidden">Copy</span>
              </CopyButton>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
