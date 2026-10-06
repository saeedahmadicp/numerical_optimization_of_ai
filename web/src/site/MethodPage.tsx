/**
 * #/method/<id> — one method's page: what it needs and promises, a live run of its first parity
 * case beside the MethodCard, the parameter table from its ParamSpecs, the Python call that
 * reproduces the run (with its real output), the parity record replayed live in the browser, and
 * its sources.
 */
import { useEffect, useMemo, useState } from 'react';
import { choiceLabel, getMethod, hasMethod, type RegisteredMethod } from '../core/registry';
import { reviveNumbers } from '../core/json';
import type { Problem, Result } from '../core/types';
import { int, paramValue, powTen } from '../core/format';
import { getProblem, hasProblem } from '../problems/registry';
import { labForFamily } from '../labs';
import { CopyButton } from '../ui/components/Copy';
import { Formula } from '../ui/components/Formula';
import { Icon } from '../ui/components/Icon';
import { MathText } from '../ui/components/MathText';
import { SciText } from '../ui/components/Num';
import { scriptsToMath } from '../ui/mathProse';
import { RateText } from '../ui/components/RateText';
import { pythonCall } from '../labs/_shell/python';
import { REPO } from '../app/catalog';
import type { CatalogMethod, CatalogProblem, ParityCase } from '../app/catalogTypes';
import { PageFrame } from '../app/Pages';
import { useCatalogIndexNow } from '../app/useCatalogIndex';
import { NotFound } from '../app/Pages';
import { loaders as parityLoaders } from 'virtual:numopt/parity';
import { LiveRun } from './LiveRun';
import { NEEDS_TEX, RATE_LABEL, rateClass } from './methodMeta';
import { caseParams, compareCase, type ParityOutcome } from './parity';
import { loadFamily, pythonSource } from './runtime';
import { CodeBlock } from './CodeBlock';
import styles from './MethodPage.module.css';

type AnyProblem = Problem<unknown> & { domain: unknown[] };

/**
 * True when the rate text already names its class ("linear (rate ½)" → Linear, "finite (vertex
 * to adjacent vertex)" → Finite or direct), so the class label would only repeat it.
 */
function repeatsClass(order: string): boolean {
  const o = order.trim().toLowerCase();
  const label = RATE_LABEL[rateClass(order)].toLowerCase();
  return !o || o.startsWith(label) || o.startsWith(label.split(/\s+/)[0]);
}

const RUN_KEYS = ['x0', 'bracket', 'seed'] as const;

/** Split a case's params into run options (x0, bracket, seed) and method parameters. */
function splitParams(params: Record<string, unknown>) {
  const options: Record<string, unknown> = {};
  const rest: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(reviveNumbers<Record<string, unknown>>(params)))
    ((RUN_KEYS as readonly string[]).includes(k) ? options : rest)[k] = v;
  return { options, rest };
}

function pyCallFor(m: CatalogMethod, c: ParityCase): string {
  const { options, rest } = splitParams(c.params);
  return pythonCall(m.id, {
    problem: c.problem,
    x0: options.x0 as number | number[] | undefined,
    bracket: options.bracket as number[] | undefined,
    seed: options.seed as number | undefined,
    params: rest as never,
    specs: m.params as never,
  });
}

function cliFor(m: CatalogMethod, c: ParityCase): string {
  const { options, rest } = splitParams(c.params);
  const parts = ['numopt', 'run', m.id, c.problem];
  const x0 = options.x0;
  if (x0 !== undefined) parts.push('--x0', ...(Array.isArray(x0) ? x0 : [x0]).map(String));
  if (Array.isArray(options.bracket)) parts.push('--bracket', ...options.bracket.map(String));
  for (const [k, v] of Object.entries(rest))
    parts.push('--set', `${k}=${Array.isArray(v) ? v.join(',') : String(v)}`);
  return parts.join(' ');
}

/** "38 iterations · converged": the count with its unit, then the verdict. */
const runText = (n: number, converged: boolean) =>
  `${int(n)} ${n === 1 ? 'iteration' : 'iterations'} · ${converged ? 'converged' : 'stopped'}`;

/** Python's repr of the snippet's last line: `(True, 33)`. */
const pyOutput = (c: ParityCase) => `(${c.converged ? 'True' : 'False'}, ${c.nIter})`;

function useParityCases(family: string | undefined, id: string): ParityCase[] | null {
  const [state, setState] = useState<{ key: string; cases: ParityCase[] } | null>(null);
  const key = `${family}:${id}`;
  useEffect(() => {
    if (!family) return;
    let alive = true;
    const load = parityLoaders[family];
    if (!load) {
      queueMicrotask(() => alive && setState({ key, cases: [] }));
      return;
    }
    void load().then((m) => {
      if (alive) setState({ key, cases: m.default.filter((c) => c.method === id) });
    });
    return () => {
      alive = false;
    };
  }, [family, id, key]);
  return state?.key === key ? state.cases : null;
}

/** Load the family's ports; then the registered method (or null when it is not ported). */
function usePort(family: string | undefined, id: string): RegisteredMethod | null | undefined {
  const [state, setState] = useState<{ key: string; m: RegisteredMethod | null } | null>(null);
  const key = `${family}:${id}`;
  useEffect(() => {
    if (!family) return;
    let alive = true;
    loadFamily(family)
      .then(() => alive && setState({ key, m: hasMethod(id) ? getMethod(id) : null }))
      .catch(() => alive && setState({ key, m: null }));
    return () => {
      alive = false;
    };
  }, [family, id, key]);
  return state?.key === key ? state.m : undefined;
}

function runCase(method: RegisteredMethod, c: ParityCase): ParityOutcome {
  const t0 = performance.now();
  try {
    const defaults = Object.fromEntries(method.spec.params.map((p) => [p.name, p.default]));
    const result = method.fn(getProblem(c.problem), caseParams(defaults, c) as never);
    const { ok, checks } = compareCase(result, c, method.spec.deterministic);
    return { ok, checks, result, ms: performance.now() - t0 };
  } catch (e) {
    return {
      ok: false,
      checks: [],
      result: null,
      error: e instanceof Error ? e.message : String(e),
      ms: performance.now() - t0,
    };
  }
}

/** Replays every case, one per frame, so the page stays responsive. */
function useLiveParity(method: RegisteredMethod | null | undefined, cases: ParityCase[] | null) {
  const [state, setState] = useState<{ key: unknown; out: ParityOutcome[] }>({
    key: null,
    out: [],
  });
  const key = method && cases ? cases : null;
  useEffect(() => {
    if (!method || !cases) return;
    let i = 0;
    let raf = 0;
    const out: ParityOutcome[] = [];
    const tick = () => {
      if (i >= cases.length) return;
      if (!hasProblem(cases[i].problem)) {
        out.push({ ok: false, checks: [], result: null, error: 'problem not ported', ms: 0 });
      } else out.push(runCase(method, cases[i]));
      i++;
      setState({ key: cases, out: [...out] });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [method, cases]);
  return state.key === key ? state.out : [];
}

/** What each `needs` entry is, in words, for screen readers (the chip shows the symbol). */
const NEEDS_SPOKEN: Record<string, string> = {
  f: 'function values f',
  grad: 'gradient ∇f',
  hess: 'Hessian ∇²f',
  jac: 'Jacobian J_F',
  grad_batch: 'sample gradients ∇f_i',
  residual: 'residuals r',
  A: 'matrix A',
  b: 'vector b',
  x0: 'start point x₀',
  bracket: 'bracket [a, b]',
  interval: 'interval [a, b]',
};

function Needs({ needs }: { needs: readonly string[] }) {
  return (
    <span className={styles.needs}>
      {needs.map((n) => (
        <span key={n} className={styles.need}>
          {NEEDS_TEX[n] ? <Formula tex={NEEDS_TEX[n]} fallback label={NEEDS_SPOKEN[n] ?? n} /> : n}
        </span>
      ))}
    </span>
  );
}

/** A parameter value as typeset text: `10⁻⁸`, `100,000`, `0.9`, `True`, `strong Wolfe`. */
function paramText(v: unknown, kind: string, log = false): string {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'number') {
    // A log range reads in powers of ten: 10⁻¹⁴ … 10⁻².
    if (log && v > 0 && Number.isInteger(Math.log10(v)) && kind !== 'int')
      return powTen(Math.log10(v));
    return paramValue(v, kind);
  }
  if (Array.isArray(v)) return `(${v.map((x) => paramText(x, kind)).join(', ')})`;
  return String(v);
}

function ParamTable({ m, port }: { m: CatalogMethod; port: RegisteredMethod | null | undefined }) {
  if (!m.params.length)
    return <p className={styles.muted}>This method has no tunable parameters.</p>;
  const ts = new Map((port?.spec.params ?? []).map((p) => [p.name, p]));
  return (
    <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Parameter table">
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Parameter</th>
            <th scope="col">Default</th>
            <th scope="col">Range</th>
            <th scope="col">Meaning</th>
          </tr>
        </thead>
        <tbody>
          {m.params.map((p) => {
            const t = ts.get(p.name);
            const choice = (c: string) => choiceLabel(t ?? {}, c);
            const range =
              p.kind === 'choice'
                ? p.choices.map((c) => choice(String(c))).join(' · ')
                : p.kind === 'bool'
                  ? 'True · False'
                  : p.min !== null || p.max !== null
                    ? `${p.min === null ? '−∞' : paramText(p.min, p.kind, p.log)} … ${p.max === null ? '∞' : paramText(p.max, p.kind, p.log)}${p.log ? ', log scale' : ''}`
                    : '—';
            const def =
              p.kind === 'choice' && typeof p.default === 'string'
                ? choice(p.default)
                : paramText(p.default, p.kind, p.log);
            return (
              <tr key={p.name}>
                <th scope="row">
                  <code className={styles.pname}>{p.name}</code>
                  {t?.tex && (
                    <span className={styles.ptex}>
                      <Formula tex={t.tex} fallback />
                    </span>
                  )}
                  <span className={styles.pkind}>{p.kind}</span>
                </th>
                <td className={styles.num}>
                  <SciText text={def} />
                </td>
                <td className={styles.num}>
                  <SciText text={range} />
                </td>
                <td className={styles.phelp}>
                  {t?.label ? <strong>{t.label}. </strong> : null}
                  {p.help}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ParityTable({
  cases,
  outcomes,
  problems,
  m,
}: {
  cases: ParityCase[];
  outcomes: ParityOutcome[];
  problems: Map<string, CatalogProblem>;
  m: CatalogMethod;
}) {
  return (
    <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Parity cases">
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Case</th>
            <th scope="col">Python</th>
            <th scope="col">TypeScript, now</th>
            <th scope="col">Checks</th>
          </tr>
        </thead>
        <tbody>
          {cases.map((c, i) => {
            const o = outcomes[i];
            const { options, rest } = splitParams(c.params);
            const args = { ...options, ...rest };
            const argText = Object.entries(args)
              .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(', ')}]` : String(v)}`)
              .join(', ');
            return (
              <tr key={i}>
                <th scope="row">
                  <span className={styles.caseName}>
                    {problems.get(c.problem)?.name ?? c.problem}
                  </span>
                  <code className={styles.caseArgs}>
                    {c.problem}
                    {argText && ` · ${argText}`}
                  </code>
                </th>
                <td className={styles.mono}>{runText(c.nIter, c.converged)}</td>
                <td className={styles.mono}>
                  {!o ? (
                    <span className={styles.muted}>running…</span>
                  ) : o.result ? (
                    <>
                      {runText(o.result.nIter, o.result.converged)}
                      <span className={styles.ms}> · {o.ms < 1 ? '<1' : Math.round(o.ms)} ms</span>
                    </>
                  ) : (
                    <span className={styles.bad}>{o.error}</span>
                  )}
                </td>
                <td>
                  {o && (
                    <ul className={styles.checks}>
                      {o.checks.map((ch) => (
                        <li key={ch.name} data-ok={ch.ok || undefined}>
                          <span aria-hidden="true">{ch.ok ? '✓' : '✗'}</span> {ch.name}{' '}
                          <span className={styles.muted}>{ch.detail}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!m.deterministic && (
        <p className={styles.note}>
          Seeded method: both languages draw from the same Mulberry32 stream, so the first ten
          iterates and the final objective are compared (libm differences in log/cos may move later
          iterates).
        </p>
      )}
    </div>
  );
}

export default function MethodPage({ id }: { id: string }) {
  const index = useCatalogIndexNow();
  const m = index.methods.find((x) => x.id === id);
  const family = m?.family;
  const lab = family ? labForFamily(family) : undefined;
  const cases = useParityCases(family, id);
  const port = usePort(family, id);
  const outcomes = useLiveParity(port, cases);
  const problems = useMemo(() => new Map(index.problems.map((p) => [p.id, p])), [index]);

  const demo = cases?.[0];
  const live = useMemo(() => {
    if (!port || !demo || !hasProblem(demo.problem)) return null;
    const out = runCase(port, demo);
    if (!out.result) return null;
    const { options, rest } = splitParams(demo.params);
    return {
      result: out.result as Result,
      problem: getProblem<AnyProblem>(demo.problem),
      call: { problem: demo.problem, options, params: rest },
    };
  }, [port, demo]);

  useEffect(() => {
    document.title = m ? `${m.name} · Methods · numopt` : 'Not found · numopt';
  }, [m]);

  if (!m) return <NotFound />;

  const siblings = index.methods.filter((x) => x.family === family);
  const pos = siblings.findIndex((x) => x.id === id);
  const prev = pos > 0 ? siblings[pos - 1] : undefined;
  const next = pos >= 0 && pos < siblings.length - 1 ? siblings[pos + 1] : undefined;
  const passed = outcomes.filter((o) => o.ok).length;
  const done = cases !== null && outcomes.length === cases.length;
  const labHref =
    lab && demo
      ? `#/lab/${lab.id}?m=${id}~0&p=${demo.problem}`
      : lab
        ? `#/lab/${lab.id}?m=${id}~0`
        : null;

  return (
    <PageFrame
      crumbs={[
        { label: 'Methods', href: '#/methods' },
        ...(lab ? [{ label: lab.title, href: `#/methods?family=${family}` }] : []),
        { label: m.name },
      ]}
      eyebrow={lab ? <a href={`#/methods?family=${family}`}>{lab.title}</a> : <span>Method</span>}
      title={m.name}
      lede={<MathText text={scriptsToMath(m.summary)} />}
    >
      {
        <>
          <dl className={styles.facts}>
            <div>
              <dt>Id</dt>
              <dd>
                <code className={styles.idCode}>{m.id}</code>
                <CopyButton text={m.id} label={`Copy the id ${m.id}`} className={styles.inlineCopy}>
                  <span className="visually-hidden">Copy</span>
                </CopyButton>
              </dd>
            </div>
            <div>
              <dt>Rate</dt>
              <dd>
                <span className={styles.rate}>
                  {m.order ? <RateText text={scriptsToMath(m.order)} /> : '—'}
                </span>
                {/* The class is a filter link, shown when the rate does not already say it. */}
                {!repeatsClass(m.order) && (
                  <a className={styles.rateClass} href={`#/methods?rate=${rateClass(m.order)}`}>
                    {RATE_LABEL[rateClass(m.order)]}
                  </a>
                )}
              </dd>
            </div>
            <div>
              <dt>Needs</dt>
              <dd>
                <Needs needs={m.needs} />
              </dd>
            </div>
            <div>
              <dt>Randomness</dt>
              <dd>{m.deterministic ? 'Deterministic' : 'Seeded (Mulberry32, shared with TS)'}</dd>
            </div>
            <div>
              <dt>Parity</dt>
              <dd>
                {cases === null || port === undefined ? (
                  <span className={styles.muted}>checking…</span>
                ) : port === null ? (
                  <span className={styles.bad}>not ported</span>
                ) : (
                  <span
                    className={styles.parityPill}
                    data-state={!done ? 'running' : passed === cases.length ? 'ok' : 'bad'}
                  >
                    {done
                      ? `${passed} of ${cases.length} ${cases.length === 1 ? 'case matches' : 'cases match'}`
                      : `${outcomes.length} of ${cases.length}…`}
                  </span>
                )}
              </dd>
            </div>
          </dl>

          <section className={styles.section} aria-labelledby="live-title">
            <div className={styles.sectionHead}>
              <h2 id="live-title" className={styles.h2}>
                Live run
              </h2>
              {labHref && (
                <a className={styles.labLink} href={labHref}>
                  Open in the {lab!.title.toLowerCase()} lab <Icon name="arrowRight" size={13} />
                </a>
              )}
            </div>
            {/*
             * Always rendered, with a reserved height (two lines; four on phones), so the figure
             * below keeps its top edge when the parity cases arrive: until then the box is empty,
             * and the sentence fills it without moving anything.
             */}
            <p className={styles.caption} aria-busy={cases === null || undefined}>
              {demo ? (
                <>
                  {m.name} on <strong>{problems.get(demo.problem)?.name ?? demo.problem}</strong>
                  {problems.get(demo.problem)?.latex && (
                    <>
                      {' '}
                      (<Formula tex={problems.get(demo.problem)!.latex} fallback />)
                    </>
                  )}
                  , the first parity case, run by the TypeScript port in your browser.
                </>
              ) : cases === null ? null : (
                'No parity case is recorded for this method, so there is no run to show.'
              )}
            </p>
            {live && port ? (
              <LiveRun
                key={id}
                method={port}
                problem={live.problem}
                result={live.result}
                family={m.family}
                call={live.call}
                order={m.order}
              />
            ) : (
              <div className={styles.live} aria-busy={port === undefined}>
                <div className={styles.livePlaceholderFigure}>
                  {port === null
                    ? 'This method has no TypeScript port yet.'
                    : port === undefined
                      ? 'Loading the TypeScript port…'
                      : 'No runnable case.'}
                </div>
                <div className={styles.livePlaceholderCard} aria-hidden="true" />
              </div>
            )}
          </section>

          <section className={styles.section} aria-labelledby="params-title">
            <h2 id="params-title" className={styles.h2}>
              Parameters
            </h2>
            <ParamTable m={m} port={port} />
          </section>

          {demo && (
            <section className={styles.section} aria-labelledby="python-title">
              <h2 id="python-title" className={styles.h2}>
                Python
              </h2>
              <p className={styles.prose}>
                The same run in the reference package. The output is the recorded result of this
                call.
              </p>
              <div className={styles.codeGrid}>
                <CodeBlock
                  label="Python"
                  lines={[
                    { prompt: '>>>', code: 'import numopt' },
                    { prompt: '>>>', code: 'from numopt import problems' },
                    { prompt: '>>>', code: `res = ${pyCallFor(m, demo)}` },
                    { prompt: '>>>', code: 'res.converged, res.n_iter' },
                  ]}
                  output={pyOutput(demo)}
                />
                <CodeBlock label="Shell" lines={[{ prompt: '$', code: cliFor(m, demo) }]} />
              </div>
              <p className={styles.sourceLine}>
                <a
                  href={`${REPO}/${pythonSource(m.family).endsWith('/') ? 'tree' : 'blob'}/main/${pythonSource(m.family)}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <Icon name="code" size={13} /> {pythonSource(m.family)}{' '}
                  <Icon name="external" size={11} />
                </a>
              </p>
            </section>
          )}

          {cases && cases.length > 0 && (
            <section className={styles.section} aria-labelledby="parity-title">
              <h2 id="parity-title" className={styles.h2}>
                Parity with Python
              </h2>
              <p className={styles.prose}>
                <code>numopt export</code> records{' '}
                {cases.length === 1 ? 'one case' : `${cases.length} cases`} for this method. The
                TypeScript port replays {cases.length === 1 ? 'it' : 'each'} here, now, and the
                result is compared exactly as <code>npm test</code> does: the first ten iterates
                within 10⁻⁸, then the iteration count, the final iterate (10⁻⁶ relative) and the
                stopping verdict.
              </p>
              <ParityTable cases={cases} outcomes={outcomes} problems={problems} m={m} />
            </section>
          )}

          {m.references.length > 0 && (
            <section className={styles.section} aria-labelledby="refs-title">
              <h2 id="refs-title" className={styles.h2}>
                Sources
              </h2>
              <ol className={styles.refs}>
                {m.references.map((r) => (
                  <li key={r}>
                    <cite>{r}</cite>
                    <CopyButton
                      text={r}
                      label={`Copy the reference: ${r}`}
                      className={styles.inlineCopy}
                    >
                      <span className="visually-hidden">Copy</span>
                    </CopyButton>
                  </li>
                ))}
              </ol>
            </section>
          )}

          <nav className={styles.pager} aria-label="Methods in this family">
            {prev ? (
              <a href={`#/method/${prev.id}`} className={styles.pagerLink}>
                <span className={styles.pagerDir}>
                  <Icon name="arrowLeft" size={12} /> Previous
                </span>
                <span className={styles.pagerName}>{prev.name}</span>
              </a>
            ) : (
              <span />
            )}
            {next && (
              <a href={`#/method/${next.id}`} className={styles.pagerLink} data-next>
                <span className={styles.pagerDir}>
                  Next <Icon name="arrowRight" size={12} />
                </span>
                <span className={styles.pagerName}>{next.name}</span>
              </a>
            )}
          </nav>
        </>
      }
    </PageFrame>
  );
}
