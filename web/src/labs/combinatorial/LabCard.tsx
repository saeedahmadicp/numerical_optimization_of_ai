/**
 * The combinatorial MethodCard: the textbook page for the current step (portal-direction §3),
 * with quantities that suit tours and selections (the shared card shows 𝐱ₖ as a vector norm,
 * which means nothing for a permutation). Head (chip · rate · menu), the rule, the rule with this
 * step's numbers, the live quantities, the stopping test that ended the run, intuition,
 * strengths and weaknesses, sources.
 */
import { useMemo } from 'react';
import type { Step } from '../../core/types';
import { defaults } from '../../core/registry';
import {
  CopyButton,
  Formula,
  Menu,
  MethodChip,
  SciText,
  type MenuItem,
} from '../../ui/components';
import { StepBlock, describeResult, evidence, type LabRun } from '../_shell';
// The shell card's status block (MethodCard), class for class: one status form in every lab.
import shell from '../_shell/blocks.module.css';
import { pythonCallFor } from './python';
import { filledRule, quantities, type DocCtx } from './docs';
import { statusResult, statusWording, type Kind } from './model';
import styles from './LabCard.module.css';

const SUB: Record<string, string> = {
  n: 'ₙ',
  i: 'ᵢ',
  j: 'ⱼ',
  k: 'ₖ',
  '0': '₀',
  '1': '₁',
  '2': '₂',
};
/** The Python messages write `z_n(C)`; show a real subscript (zₙ(C)), never a raw underscore. */
const subscripts = (t: string | null) =>
  t && t.replace(/([A-Za-z])_([nijk012])(?![A-Za-z0-9])/g, (_, a: string, b: string) => a + SUB[b]);

/** Numbers do not change a formula's height; its shape (fractions, rows, operators) does. */
const texShape = (tex: string) => tex.replace(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi, '0');

/**
 * The longest filled rule of each shape over the run. Laid out invisibly with the current one,
 * they keep the step block at one height for the whole run (no layout shift during playback).
 */
function ruleSizers(id: string, trace: readonly Step[], ctx: DocCtx): string[] {
  const byShape = new Map<string, string>();
  for (const st of trace) {
    const t = filledRule(id, st, ctx);
    if (!t) continue;
    const key = texShape(t);
    const cur = byShape.get(key);
    if (!cur || t.length > cur.length) byShape.set(key, t);
  }
  return [...byShape.values()];
}

export function LabCard({
  run,
  step,
  ctx,
  kind,
  seed,
}: {
  run: LabRun;
  step: Step | undefined;
  ctx: DocCtx;
  kind: Kind;
  seed: number | undefined;
}) {
  const { spec, doc } = run.method;
  const filled = filledRule(spec.id, step, ctx);
  const sizers = useMemo(
    () => ruleSizers(spec.id, run.result.trace, ctx),
    [spec.id, run.result.trace, ctx],
  );
  const live = quantities(spec.id, step, ctx, kind);
  const fitGroup = `comb-step~${spec.id}~${run.sel.slot}~${ctx.problem.id}~${run.result.trace.length}`;
  const status = describeResult(
    statusResult(spec.id, run.result),
    run.error,
    statusWording(spec.id),
  );
  const params = { ...defaults(spec), ...run.sel.params };
  // The sentence already ends "— 2-optimal": the evidence drops a leading "2-optimal: ".
  const qualifier = statusWording(spec.id).converged?.long;
  const raw = run.error ? null : subscripts(evidence(run.result.message, params));
  const why =
    raw && qualifier && status.icon === '✓' && raw.startsWith(`${qualifier}: `)
      ? raw.slice(qualifier.length + 2)
      : raw;
  // Greedy: the ½ guarantee holds only with the single-item fix.
  const rate =
    spec.id === 'knapsack_greedy'
      ? params.single_item_fix === false
        ? 'heuristic · O(n log n), no guarantee'
        : 'heuristic · ½-approximation'
      : (doc?.order ?? spec.order);
  const menu: MenuItem[] = [
    {
      kind: 'copy',
      label: 'Copy Python call',
      icon: 'code',
      text: () =>
        pythonCallFor(
          spec.id,
          ctx.problem,
          run.sel.params,
          spec.deterministic ? undefined : seed,
          spec.params,
        ),
    },
    ...(doc?.rule
      ? [{ kind: 'copy' as const, label: 'Copy update rule (LaTeX)', text: doc.rule }]
      : []),
    ...(filled ? [{ kind: 'copy' as const, label: 'Copy this step (LaTeX)', text: filled }] : []),
    ...(step
      ? [
          {
            kind: 'copy' as const,
            label: kind === 'tsp' ? 'Copy the current tour' : 'Copy the current selection',
            text: () => `[${(step.x as number[]).join(', ')}]`,
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
  return (
    <div className={styles.card} tabIndex={0} role="region" aria-label={`${spec.name} details`}>
      <div className={styles.head}>
        <MethodChip name={spec.name} slot={run.sel.slot} />
        <span className={styles.headEnd}>
          {rate && (
            <span className={styles.rate} title={spec.order}>
              {rate}
            </span>
          )}
          <Menu label={`More actions for ${spec.name}`} items={menu} />
        </span>
      </div>
      {doc?.rule && (
        <div className={styles.rule}>
          <Formula tex={doc.rule} display fit />
        </div>
      )}
      {sizers.length > 0 && (
        // Step k's rule is the move from k − 1 to k ("Step 46 → 47"); the sizers keep its height.
        <StepBlock
          k={Math.max(0, (step?.k ?? 0) - 1)}
          label={(step?.k ?? 0) === 0 ? 'Start' : undefined}
          reset={`${spec.id}|${run.sel.slot}`}
        >
          <div className={styles.filledStack}>
            {/* One fit scale for the step and its sizers (the longest set it), so the step
                is never set larger, hence taller, than the sizers that reserve its height. */}
            <div>{filled && <Formula tex={filled} display fit fitGroup={fitGroup} />}</div>
            {sizers.map((t) => (
              <div key={t} className={styles.sizer} aria-hidden="true">
                <Formula tex={t} display fit fitGroup={fitGroup} />
              </div>
            ))}
          </div>
        </StepBlock>
      )}
      {live.length > 0 && (
        <dl className={styles.live} aria-label="This step's quantities">
          {live.map((q) => (
            <div key={q.tex} className={styles.liveCell} title={q.label}>
              <dt className={styles.liveLabel}>
                <Formula tex={q.tex} />
              </dt>
              <dd className={styles.liveValue}>{q.value}</dd>
            </div>
          ))}
        </dl>
      )}
      <div className={shell.statusBlock}>
        <p className={shell.status} data-tone={status.tone}>
          <span className={shell.statusIcon} aria-hidden="true">
            {status.icon}
          </span>
          <span>
            <SciText text={status.long} />
          </span>
        </p>
        {why && (status.tone === 'good' || status.icon === '◷') && (
          <p className={shell.evidence}>{why[0].toUpperCase() + why.slice(1)}.</p>
        )}
      </div>
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
    </div>
  );
}
