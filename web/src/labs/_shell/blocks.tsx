/** Building blocks for a lab's control rail and insights. Compose them inside <LabShell>. */
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import type { RegisteredMethod } from '../../core/registry';
import { defaults } from '../../core/registry';
import type { ParamValue, Params, Result, RunOptions, Step } from '../../core/types';
import type { MethodSelection } from './slots';
import { int, sci, sig, vec } from '../../core/format';
import { norm } from '../../core/linalg';
import { firstFreeSlot } from './slots';
import {
  describeResult,
  evidence,
  methodWord,
  shortMethodName,
  type IterationNoun,
  type RunStatus,
  type StatusWording,
} from './status';
import { pythonCall } from './python';
import { liveSpans } from './liveGrid';
import {
  Badge,
  CopyButton,
  Formula,
  FormulaScroll,
  IconButton,
  MathText,
  Menu,
  Icon,
  MethodChip,
  NumberField,
  ParamControls,
  RateText,
  SciText,
  SegmentedControl,
  Select,
  Swatch,
  type MenuItem,
  type SelectOption,
} from '../../ui/components';
import { useMotionTransitions } from '../../ui/motion';
import { mathTextToPlain, scriptsToMath } from '../../ui/mathProse';
import { seriesVar } from '../../ui/colors';
import styles from './blocks.module.css';

// ── Problem picker ─────────────────────────────────────────────────────────────────────

export interface ProblemOption {
  id: string;
  name: string;
  latex: string;
  description?: string;
  tags?: string[];
}

export function ProblemPicker({
  problems,
  value,
  onChange,
}: {
  problems: readonly ProblemOption[];
  value: string;
  onChange: (id: string) => void;
}) {
  const current = problems.find((p) => p.id === value);
  const options: SelectOption[] = problems.map((p) => ({
    value: p.id,
    label: p.name,
    description: p.description,
    keywords: p.tags?.join(' '),
  }));
  return (
    <>
      <Select
        label="Problem"
        value={value}
        options={options}
        onChange={onChange}
        searchable={problems.length > 5}
      />
      {current && (
        <>
          <div className={styles.formulaBox}>
            <FormulaScroll>
              <Formula tex={current.latex} display fit />
            </FormulaScroll>
          </div>
          {current.description && (
            <p className={styles.desc}>
              <MathText text={scriptsToMath(current.description)} />
            </p>
          )}
        </>
      )}
    </>
  );
}

// ── Method slots ───────────────────────────────────────────────────────────────────────

export type { MethodSelection } from './slots';

export interface MethodSlotsProps {
  available: readonly RegisteredMethod[];
  value: readonly MethodSelection[];
  onChange: (v: MethodSelection[]) => void;
  /**
   * Params the lab controls elsewhere (hidden here): one list for every method, or a list per
   * method id (`{ exact_quadratic: ['c1'] }`).
   */
  hiddenParams?: readonly string[] | Readonly<Record<string, readonly string[]>>;
  /**
   * Let the same method be added twice with different params (damped and undamped Newton side
   * by side). Pass the same flag to `useMethodSelection`. Entries are then keyed by color slot.
   */
  allowDuplicates?: boolean;
}

/** Up to four color-coded methods, each with generated parameter controls. */
export function MethodSlots({
  available,
  value,
  onChange,
  hiddenParams,
  allowDuplicates = false,
}: MethodSlotsProps) {
  // The open panel is identified by color slot (unique even when a method appears twice).
  const [open, setOpen] = useState<number | null>(value[0]?.slot ?? null);
  const motionT = useMotionTransitions();
  const byId = new Map(available.map((m) => [m.spec.id, m]));
  const free = firstFreeSlot(value);
  const addable: SelectOption[] = available
    .filter((m) => allowDuplicates || !value.some((v) => v.id === m.spec.id))
    .map((m) => ({ value: m.spec.id, label: m.spec.name, description: m.spec.summary }));
  const hiddenFor = (id: string): readonly string[] | undefined =>
    hiddenParams === undefined
      ? undefined
      : Array.isArray(hiddenParams)
        ? (hiddenParams as readonly string[])
        : (hiddenParams as Readonly<Record<string, readonly string[]>>)[id];

  const setParam = (slot: number, name: string, v: ParamValue) =>
    onChange(
      value.map((m) => (m.slot === slot ? { ...m, params: { ...m.params, [name]: v } } : m)),
    );

  return (
    <div className={styles.slots}>
      <AnimatePresence initial={false}>
        {value.map((sel) => {
          const m = byId.get(sel.id);
          if (!m) return null;
          const expanded = open === sel.slot;
          return (
            <motion.div
              key={`${sel.id}~${sel.slot}`}
              layout={!motionT.reduced}
              className={styles.slot}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: motionT.exit }}
              transition={motionT.base}
            >
              {/*
               * Name (up to two lines, never cut at desktop width) · Params toggle · remove, each
               * in its own column; the head keeps a two-line height for every method.
               */}
              <div className={styles.slotHead}>
                <MethodChip name={m.spec.name} slot={sel.slot} lines={2} />
                {m.spec.params.length > 0 ? (
                  <button
                    type="button"
                    className={styles.slotToggle}
                    aria-expanded={expanded}
                    aria-label={`${expanded ? 'Hide' : 'Show'} ${m.spec.name} parameters`}
                    onClick={() => setOpen(expanded ? null : sel.slot)}
                  >
                    Params
                    <Icon name="chevronDown" size={12} />
                  </button>
                ) : (
                  <span />
                )}
                {value.length > 1 ? (
                  <IconButton
                    size="sm"
                    icon="x"
                    label={`Remove ${m.spec.name}`}
                    className={styles.slotRemove}
                    onClick={() => onChange(value.filter((v) => v.slot !== sel.slot))}
                  />
                ) : (
                  <span />
                )}
              </div>
              <AnimatePresence initial={false}>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0, transition: motionT.exit }}
                    transition={motionT.base}
                    style={{ overflow: 'hidden' }}
                  >
                    <div className={styles.slotBody}>
                      {m.spec.summary && (
                        <p className={styles.slotSummary}>
                          <MathText text={scriptsToMath(m.spec.summary)} />
                        </p>
                      )}
                      <ParamControls
                        specs={m.spec.params}
                        values={{ ...defaults(m.spec), ...sel.params }}
                        onChange={(name, v) => setParam(sel.slot, name, v)}
                        labelPrefix={m.spec.name}
                        hidden={hiddenFor(sel.id)}
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </AnimatePresence>
      {free >= 0 && addable.length > 0 && (
        <Select
          label="Add a method"
          value={null}
          placeholder="Add a method to compare…"
          options={addable}
          renderValue={() => (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                color: 'var(--color-text-3)',
              }}
            >
              <Swatch slot={free} />
              Add a method to compare…
            </span>
          )}
          onChange={(id) => {
            onChange([...value, { id, slot: free, params: {} }]);
            setOpen(free);
          }}
        />
      )}
    </div>
  );
}

// ── Start point ────────────────────────────────────────────────────────────────────────

/**
 * The one wording of "click the plot to set the start point", with the crosshair icon: "Click
 * the plot to set 𝐱₀" (and "· drag 𝐱₀ to move it" in labs where the start marker can be dragged).
 * StartPointFields shows it under the fields; LabShell's `stageHint` shows it on the plot.
 */
export function StartPointHint({
  variable = '𝐱₀',
  drag = false,
  className,
}: {
  /** The start point's symbol as plain text: '𝐱₀' (a vector), 'x₀' (a scalar), '𝐰₀'. */
  variable?: string;
  /** The lab lets the viewer drag the start marker. */
  drag?: boolean;
  className?: string;
}) {
  return (
    <span className={`${styles.hintText} ${className ?? ''}`}>
      <Icon name="crosshair" size={13} />
      <span>
        Click the plot to set {variable}
        {drag && ` · drag ${variable} to move it`}
      </span>
    </span>
  );
}

export function StartPointFields({
  value,
  onChange,
  variable,
  drag,
  hint,
}: {
  value: readonly [number, number] | readonly number[];
  onChange: (v: [number, number]) => void;
  /** The start point's symbol in the shared hint ('𝐱₀' by default). */
  variable?: string;
  /** The start marker can be dragged (adds "· drag 𝐱₀ to move it"). */
  drag?: boolean;
  /**
   * @deprecated Pass `variable` (and `drag`) so every lab says the same thing; a custom hint is
   * shown as given.
   */
  hint?: ReactNode;
}) {
  return (
    <>
      <div className={styles.xy}>
        <NumberField
          label="Start x"
          prefix="x₀"
          value={value[0]}
          onChange={(v) => onChange([v, value[1]])}
        />
        <NumberField
          label="Start y"
          prefix="y₀"
          value={value[1]}
          onChange={(v) => onChange([value[0], v])}
        />
      </div>
      <p className={styles.hint}>
        {hint ? (
          <span className={styles.hintText}>
            <Icon name="crosshair" size={13} />
            <span>{hint}</span>
          </span>
        ) : (
          <StartPointHint variable={variable} drag={drag} />
        )}
      </p>
    </>
  );
}

// ── Method card ────────────────────────────────────────────────────────────────────────

function readKey(step: Step, key: string): unknown {
  if (key.startsWith('info.')) return step.info[key.slice(5)];
  return (step as unknown as Record<string, unknown>)[key];
}

/**
 * 4–6 significant digits (`digits` overrides); scientific below 10⁻³ and above 10⁵; vectors up
 * to 3 entries.
 */
function show(v: unknown, digits?: number): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number')
    return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5
      ? sci(v, digits ? Math.max(1, digits - 1) : 3)
      : sig(v, digits ?? 5);
  if (Array.isArray(v) && v.every((x) => typeof x === 'number'))
    return v.length <= 3
      ? vec(v as number[], digits ?? 4)
      : `‖·‖ = ${sig(norm(v as number[]), digits ?? 4)}`;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  return String(v);
}

const RUN_KEYS = new WeakMap<object, number>();
let runCounter = 0;
/** A stable key per Result object (a new run gets a new key). */
function runKey(result: Result | undefined): number {
  if (!result) return 0;
  let k = RUN_KEYS.get(result);
  if (k === undefined) RUN_KEYS.set(result, (k = ++runCounter));
  return k;
}

/**
 * A block that never shrinks while `reset` stays the same: it remembers the tallest height its
 * content has had (a rule filled in step by step, the "this step" numbers) and keeps it as its
 * min-height, so the blocks below it do not move from step to step. A new `reset` starts over.
 */
function KeepHeight({
  className,
  reset,
  children,
  ...rest
}: {
  className?: string;
  reset: string;
  children: ReactNode;
  'aria-label'?: string;
  role?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.minHeight = '';
    let max = 0;
    const measure = () => {
      const h = el.getBoundingClientRect().height;
      if (h > max + 0.5) {
        max = h;
        el.style.minHeight = `${Math.ceil(max)}px`;
      }
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    // Width changes reflow the content: start over at the new width.
    let width = el.clientWidth;
    const rw = new ResizeObserver(() => {
      if (el.clientWidth === width) return;
      width = el.clientWidth;
      max = 0;
      el.style.minHeight = '';
      measure();
    });
    rw.observe(el);
    return () => {
      ro.disconnect();
      rw.disconnect();
    };
  }, [reset]);
  return (
    <div ref={ref} className={className} {...rest}>
      {children}
    </div>
  );
}

export interface MethodCardProps {
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
  /**
   * Replaces `doc.quantities` (e.g. a lab with textbook indexing relabels or drops rows). An
   * entry with `key: 'x'` or `key: 'fun'` replaces the default 𝐱ₖ or f(𝐱ₖ) cell, `omit` drops
   * it; `digits` sets significant digits.
   */
  quantities?: readonly MethodCardQuantity[];
  /** What one step is called in the status sentence (`['trial', 'trials']`). */
  iterationNoun?: IterationNoun;
  /** The lab's status wording (unit and convergence qualifier); overrides `iterationNoun`. */
  wording?: StatusWording;
  /**
   * The lab's own "this step" block (a step equation, a quadrature readout). It takes the place
   * of the live-quantities grid, after the update rule, so every lab's card has one order:
   * chip and rate, rule, this step, status, prose, sources.
   */
  live?: ReactNode;
}

export interface MethodCardQuantity {
  tex: string;
  /**
   * `stepSize`, `x`, `fun`, … or `info.<key>`. An entry with `key: 'x'` or `key: 'fun'` relabels
   * the 𝐱ₖ or f(𝐱ₖ) cell (an integral estimate, a tour length); with `omit` it removes it.
   */
  key: string;
  label?: string;
  digits?: number;
  /** Remove this cell (e.g. no 𝐱ₖ cell for a permutation). */
  omit?: boolean;
}

/**
 * The lab's textbook page for the current step (portal-direction §3): head (chip · rate · menu),
 * the update rule between hairlines, this step's numbers, the status with its evidence, the
 * intuition, strengths and weaknesses, and the sources in italic with a copy action.
 */
export function MethodCard({
  method,
  slot,
  step,
  result,
  error,
  call,
  quantities: quantitiesOverride,
  iterationNoun,
  wording,
  live: liveSlot,
}: MethodCardProps) {
  const { spec, doc } = method;
  const order = doc?.order ?? spec.order;
  const idBase = `card-${spec.id}-${slot}`;
  // One fit and one reserved height per run: a rule filled in step by step keeps its size.
  const ruleGroup = `${spec.id}~${slot}~${runKey(result)}`;
  const vector = Array.isArray(step?.x);
  const xTex = vector ? '\\mathbf{x}_k' : 'x_k';
  const all: readonly MethodCardQuantity[] = quantitiesOverride ?? doc?.quantities ?? [];
  const xCell = all.find((q) => q.key === 'x');
  const funCell = all.find((q) => q.key === 'fun');
  const quantities = all.filter((q) => q.key !== 'x' && q.key !== 'fun' && !q.omit);
  // One set of cells for the whole run (a value a step lacks shows "—"), so no cell appears or
  // disappears between steps and the grid never reflows during playback.
  const given = (v: unknown) => v !== null && v !== undefined;
  const trace = result?.trace ?? [];
  const hasGrad = trace.length ? trace.some((t) => given(t.gradNorm)) : given(step?.gradNorm);
  const hasStep = trace.length ? trace.some((t) => given(t.stepSize)) : given(step?.stepSize);
  // An array of 2+ entries ([a_k, b_k], (x_l, x_m, x_r)) gets a full-width cell, never an ellipsis.
  const wide = (v: unknown) => Array.isArray(v) && v.length > 1;
  const live: { tex: string; value: string; wide?: boolean }[] = step
    ? [
        { tex: 'k', value: int(step.k) },
        ...(funCell?.omit
          ? []
          : [
              funCell
                ? { tex: funCell.tex, value: show(step.fun, funCell.digits) }
                : { tex: `f(${xTex})`, value: show(step.fun) },
            ]),
        ...(hasGrad ? [{ tex: `\\|\\nabla f(${xTex})\\|`, value: show(step.gradNorm) }] : []),
        ...(xCell?.omit
          ? []
          : [
              xCell
                ? { tex: xCell.tex, value: show(step.x, xCell.digits), wide: wide(step.x) }
                : {
                    tex: xTex,
                    value: show(step.x),
                    wide: vector && (step.x as number[]).length > 1,
                  },
            ]),
        ...(hasStep && !all.some((q) => q.key === 'stepSize')
          ? [{ tex: '\\alpha_k', value: show(step.stepSize) }]
          : []),
        ...quantities.map((q) => {
          const v = readKey(step, q.key);
          return { tex: q.tex, value: show(v, q.digits), wide: wide(v) };
        }),
      ]
    : [];
  const spans = liveSpans(live.map((q) => Boolean(q.wide)));
  const status = result ? describeResult(result, error, wording ?? iterationNoun) : null;
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
      label: 'Open the method page',
      icon: 'book' as const,
      href: `#/method/${encodeURIComponent(spec.id)}`,
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
        <KeepHeight className={styles.rule} reset={ruleGroup}>
          <Formula tex={doc.rule} display fit fitGroup={ruleGroup} />
        </KeepHeight>
      )}
      {liveSlot}
      {!liveSlot && live.length > 0 && (
        <dl className={styles.live} aria-label="This step">
          {live.map((q, i) => (
            <div
              key={q.tex}
              className={styles.liveCell}
              style={spans[i] > 1 ? { gridColumn: `span ${spans[i]}` } : undefined}
            >
              <dt className={styles.liveLabel}>
                <Formula tex={q.tex} />
              </dt>
              <dd className={styles.liveValue}>
                <SciText text={q.value} />
              </dd>
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
            <span>
              <SciText text={status.long} />
            </span>
          </p>
          {why && (status.tone === 'good' || status.icon === '◷') && (
            <p className={styles.evidence}>
              <SciText text={why} />
            </p>
          )}
        </div>
      )}
      {(doc?.intuition || spec.summary) && (
        <p className={styles.intuition}>
          <MathText text={scriptsToMath(doc?.intuition ?? spec.summary)} />
        </p>
      )}
      {(doc?.pros?.length || doc?.cons?.length) && (
        <div className={styles.proscons}>
          {doc?.pros?.length ? (
            <div className={styles.pro}>
              <p className={styles.prosHead} id={`${idBase}-pros`}>
                Strengths
              </p>
              <ul aria-labelledby={`${idBase}-pros`}>
                {doc.pros.map((p) => (
                  <li key={p}>
                    <MathText text={scriptsToMath(p)} />
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div />
          )}
          {doc?.cons?.length ? (
            <div className={styles.con}>
              <p className={styles.prosHead} id={`${idBase}-cons`}>
                Weaknesses
              </p>
              <ul aria-labelledby={`${idBase}-cons`}>
                {doc.cons.map((p) => (
                  <li key={p}>
                    <MathText text={scriptsToMath(p)} />
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      )}
      {spec.references.length > 0 && (
        <div className={styles.refs} role="group" aria-label="Sources">
          {spec.references.map((r) => (
            <p key={r} className={styles.ref}>
              <cite>
                <MathText text={scriptsToMath(r)} />
              </cite>
              <CopyButton
                text={r}
                label={`Copy the reference: ${mathTextToPlain(r)}`}
                className={styles.refCopy}
              >
                <span className="visually-hidden">Copy</span>
              </CopyButton>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// ── This step ──────────────────────────────────────────────────────────────────────────

/**
 * The "this step" block of every lab, placed through MethodCard's `live` prop (after the rule):
 * sunken background, a caps label in one fixed format ("Step 3 → 4", or "Last iterate" at the
 * end), and a height that never shrinks within a run (`reset` starts it over, e.g. on a new run).
 *
 *   <MethodCard … live={<StepBlock k={k} last={k === n - 1} reset={runId}>…</StepBlock>} />
 */
export function StepBlock({
  k,
  last = false,
  label,
  reset = '',
  children,
}: {
  /** The step shown: from 𝐱ₖ to 𝐱ₖ₊₁. */
  k: number;
  /** The run's last iterate (no next step): the label reads "Last iterate". */
  last?: boolean;
  /** Overrides the label (keep the format: "Step k → k+1"). */
  label?: ReactNode;
  /** Changing this lets the reserved height shrink again (a new run, a new method). */
  reset?: string;
  children: ReactNode;
}) {
  const text = label ?? (last ? 'Last iterate' : `Step ${int(k)} → ${int(k + 1)}`);
  return (
    <KeepHeight className={styles.stepBlock} reset={reset} role="group" aria-label="This step">
      <p className={styles.stepLabel}>{text}</p>
      {children}
    </KeepHeight>
  );
}

// ── Focus picker ───────────────────────────────────────────────────────────────────────

/** A run the focus picker can show: `{ value, slot, name }`. */
export interface FocusRun {
  /** The value the picker reports (usually the method id). */
  value: string;
  slot: number;
  name: string;
}

/**
 * "Method shown in detail": one swatch per run (the name in the tooltip and the accessible
 * name). LabShell renders it in the details insight's header when given `focus`.
 */
export function FocusPicker({
  runs,
  value,
  onChange,
}: {
  runs: readonly FocusRun[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <SegmentedControl
      label="Method shown in detail"
      value={value}
      onChange={onChange}
      options={runs.map((r) => ({
        value: r.value,
        label: <Swatch slot={r.slot} size={9} />,
        ariaLabel: r.name,
        tooltip: r.name,
      }))}
    />
  );
}

// ── Iteration axis ─────────────────────────────────────────────────────────────────────

/**
 * Linear k or log k for a convergence chart, placed first in the convergence insight's
 * `actions`. Pair it with `useKAxis(lengths)` (labState.ts): one URL key (`kx`) for every lab,
 * and "auto" (the default) picks log k when the runs differ ≥ 30× in length (`suggestLogK`).
 */
export function KAxisControl({
  logX,
  onChange,
}: {
  /** The resolved axis (auto already applied). */
  logX: boolean;
  onChange: (mode: 'lin' | 'log') => void;
}) {
  return (
    <SegmentedControl
      label="Iteration axis"
      value={logX ? 'log' : 'lin'}
      onChange={onChange}
      options={[
        {
          value: 'lin',
          label: <Formula tex="k" />,
          ariaLabel: 'Linear k axis',
          tooltip: 'Linear iteration axis',
        },
        {
          value: 'log',
          label: <Formula tex="\log k" />,
          ariaLabel: 'Logarithmic k axis',
          tooltip:
            'Logarithmic iteration axis (k ≥ 1): runs of very different lengths side by side',
        },
      ]}
    />
  );
}

// ── Run summary ────────────────────────────────────────────────────────────────────────

export interface RunSummaryItem {
  name: string;
  slot: number;
  result: Result;
  error?: string;
  /** This run's status wording, when it differs from the summary's (`wording`). */
  wording?: StatusWording;
}

/**
 * The status badge of one run: the glyph, an optional lab qualifier and the count ("✓ 38",
 * "✓ 2-opt 47"), in the status tone, with tabular figures and a fixed minimum width so a count
 * that changes during playback does not move what follows it. `children` replaces the glyph and
 * count with a domain value (h⋆, R²) in the same badge.
 */
export function RunBadge({ status, children }: { status: RunStatus; children?: ReactNode }) {
  return (
    <span className={styles.summaryBadge}>
      <Badge tone={status.tone}>
        {children ?? (
          <span aria-hidden="true">
            {status.icon}
            {status.qualifier ? ` ${status.qualifier}` : ''} {status.count}
          </span>
        )}
      </Badge>
    </span>
  );
}

export interface RunSummaryProps {
  runs: readonly RunSummaryItem[];
  /** What one step is called (`['trial', 'trials']`), for the full status. */
  iterationNoun?: IterationNoun;
  /** The lab's status wording (unit and convergence qualifier); overrides `iterationNoun`. */
  wording?: StatusWording;
  /**
   * A domain value in place of the glyph and count ("h⋆ = 10⁻⁵", "R² = 0.981"), rendered inside
   * the shared badge (tone, size, minimum width). Return `undefined` to keep the default badge.
   * The full status stays the item's accessible text and tooltip.
   */
  badge?: (run: RunSummaryItem, status: RunStatus) => ReactNode;
}

/**
 * The stage header's run summary, the one form for every lab: per run a chip (swatch and the
 * short name — no parenthetical, at most 14ch with an ellipsis) and a status badge. It stays on
 * one row, so the stage head keeps its height. When the chips do not fit the row, every chip
 * first shortens its name to the first word ("Conjugate", "Newton"), then drops the name and keeps
 * the swatch, so every run and every badge stays visible (the full name and status are the
 * tooltip and the accessible text). Hidden when the stage is narrower than 560 px (the
 * convergence legend names each series).
 */
export function RunSummary({ runs, iterationNoun, wording, badge }: RunSummaryProps) {
  const box = useRef<HTMLDivElement>(null);
  const fullProbe = useRef<HTMLDivElement>(null);
  const wordProbe = useRef<HTMLDivElement>(null);
  // Compare the width the chips need with names and with first words (invisible copies) with the
  // width the row has. The answer is written to the DOM (no re-render), before paint and on
  // every resize: '' (names), 'word', or 'swatch'.
  useLayoutEffect(() => {
    const el = box.current;
    const full = fullProbe.current;
    const word = wordProbe.current;
    if (!el || !full || !word) return;
    const update = () => {
      const room = el.clientWidth + 0.5;
      const mode = full.offsetWidth <= room ? null : word.offsetWidth <= room ? 'word' : 'swatch';
      if (mode) el.setAttribute('data-compact', mode);
      else el.removeAttribute('data-compact');
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    ro.observe(full);
    ro.observe(word);
    return () => ro.disconnect();
  }, []);
  const items = runs.map((r) => {
    const st = describeResult(r.result, r.error, r.wording ?? wording ?? iterationNoun);
    return { r, st, custom: badge?.(r, st) };
  });
  const chips = (probe: false | 'full' | 'word') =>
    items.map(({ r, st, custom }) => (
      <span
        key={`${r.name}~${r.slot}`}
        className={styles.summaryItem}
        role={probe ? undefined : 'listitem'}
        title={probe ? undefined : `${r.name}: ${st.long}`}
      >
        <span
          className={styles.summaryChip}
          style={{ ['--_c' as string]: seriesVar(r.slot) }}
          aria-hidden="true"
        >
          <span className={styles.summarySwatch} />
          {probe !== 'word' && (
            <span className={styles.summaryName}>{shortMethodName(r.name)}</span>
          )}
          {probe !== 'full' && <span className={styles.summaryWord}>{methodWord(r.name)}</span>}
        </span>
        <RunBadge status={st}>{custom}</RunBadge>
        {!probe && <span className="visually-hidden">{`${r.name}: ${st.long}`}</span>}
      </span>
    ));
  return (
    <div ref={box} className={styles.summary} role="list" aria-label="Runs">
      {chips(false)}
      <div ref={fullProbe} className={styles.summaryProbe} aria-hidden="true">
        {chips('full')}
      </div>
      <div ref={wordProbe} className={styles.summaryProbe} aria-hidden="true">
        {chips('word')}
      </div>
    </div>
  );
}
