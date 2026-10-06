/**
 * #/python — getting started with the reference package: install, the first call, the Result and
 * Step records, the command line, and how the portal's TypeScript ports are held to the Python
 * record. Every output on this page is the package's real output.
 */
import { int } from '../core/format';
import { Icon } from '../ui/components/Icon';
import { CATALOG, REPO } from '../app/catalog';
import { PageFrame } from '../app/Pages';
import { useCatalogIndexNow } from '../app/useCatalogIndex';
import { CodeBlock } from './CodeBlock';
import styles from './Python.module.css';

const ARCH = `${REPO}/blob/main/docs/architecture.md`;

const SECTIONS = [
  ['install', 'Install'],
  ['first', 'A first run'],
  ['call', 'The call'],
  ['records', 'Result and Step'],
  ['cli', 'The command line'],
  ['parity', 'How parity works'],
  ['more', 'Further reading'],
] as const;

const RESULT_FIELDS: [string, string, string][] = [
  ['method', 'str', 'The method id.'],
  ['x', 'float | ndarray', 'The final iterate (or the answer: a root, an integral, a tour).'],
  [
    'fun',
    'float | None',
    'f(x) for minimization, the residual for roots, the estimate for quadrature.',
  ],
  [
    'converged',
    'bool',
    'True only when the documented tolerance test passed. Max-iter, non-finite values, singular systems and lost brackets give False.',
  ],
  ['message', 'str', 'Why the run stopped, with the numbers: "‖∇f‖∞ = 1.31e-11 ≤ gtol".'],
  ['n_iter', 'int', 'Iterations; equals trace[-1].k for iterative methods.'],
  ['n_fev · n_gev · n_hev', 'int', 'Exact counts of f, gradient and Hessian evaluations.'],
  ['trace', 'list[Step]', 'One Step for k = 0 (the start) and one per iteration.'],
  ['extra', 'dict', 'Method-specific results (tableaux, tours, interpolants).'],
];

const STEP_FIELDS: [string, string, string][] = [
  ['k', 'int', 'Iteration index; 0 is the state before any update.'],
  ['x', 'float | ndarray', 'The current iterate.'],
  ['fun', 'float | None', 'The objective, residual or estimate at x.'],
  ['grad_norm', 'float | None', '‖∇f(x)‖₂ when the method has it.'],
  [
    'step_size',
    'float | None',
    'The step length, trust radius or spacing that produced this iterate.',
  ],
  [
    'info',
    'dict',
    'The geometry of the iteration, documented per module: bracket, direction, trials, simplex, radius, tableau, tour, H, …',
  ],
];

function Fields({ rows, caption }: { rows: [string, string, string][]; caption: string }) {
  return (
    <div className={styles.tableWrap} tabIndex={0} role="region" aria-label={caption}>
      {/* Explicit roles: the phone layout restyles the rows as grids, which can drop the
          implicit table semantics in some browsers. */}
      <table className={styles.table} role="table">
        <caption className="visually-hidden">{caption}</caption>
        <thead role="rowgroup">
          <tr role="row">
            <th scope="col" role="columnheader">
              Field
            </th>
            <th scope="col" role="columnheader">
              Type
            </th>
            <th scope="col" role="columnheader">
              Meaning
            </th>
          </tr>
        </thead>
        <tbody role="rowgroup">
          {rows.map(([f, t, m]) => (
            <tr key={f} role="row">
              <th scope="row" role="rowheader">
                <code>{f}</code>
              </th>
              <td className={styles.type} role="cell">
                {t}
              </td>
              <td role="cell">{m}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function PythonPage() {
  const index = useCatalogIndexNow();
  const cases = index.methods.reduce((n, m) => n + m.cases, 0);
  return (
    <PageFrame
      title="Python"
      lede={`numopt is a lean, typed Python package (NumPy only): ${int(CATALOG.methods)} methods in ${CATALOG.families} families, ${int(CATALOG.problems)} test problems with exact derivatives, a command line, and the exporter that writes the records this portal is tested against.`}
    >
      <div className={styles.layout}>
        <nav className={styles.toc} aria-label="On this page">
          <ol>
            {SECTIONS.map(([id, title]) => (
              <li key={id}>
                <a
                  href={`#/python`}
                  onClick={(e) => {
                    e.preventDefault();
                    const el = document.getElementById(id);
                    el?.scrollIntoView({ block: 'start' });
                    el?.focus({ preventScroll: true });
                  }}
                >
                  {title}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <div className={styles.body}>
          <section aria-labelledby="install">
            <h2 id="install" className={styles.h2} tabIndex={-1}>
              Install
            </h2>
            <p className={styles.prose}>
              Python 3.11 or newer; the only runtime dependency is NumPy. The package is installed
              from GitHub until its PyPI name is settled (<code>numopt</code> on PyPI is another
              project).
            </p>
            <CodeBlock
              label="Shell"
              lines={[
                { prompt: '$', code: `pip install git+${REPO}` },
                { prompt: '', code: '# or, to run the tests and the research studies:' },
                { prompt: '$', code: `git clone ${REPO}.git && cd numerical_optimization_of_ai` },
                { prompt: '$', code: 'pip install -e ".[dev]"' },
              ]}
            />
          </section>

          <section aria-labelledby="first">
            <h2 id="first" className={styles.h2} tabIndex={-1}>
              A first run
            </h2>
            <p className={styles.prose}>
              BFGS on Rosenbrock's function from the classic start. The output is the package's.
            </p>
            <CodeBlock
              label="Python"
              lines={[
                { prompt: '>>>', code: 'import numopt' },
                { prompt: '>>>', code: 'from numopt import problems' },
                {
                  prompt: '>>>',
                  code: 'res = numopt.run("bfgs", problems.get("rosenbrock"), x0=[-1.2, 1.0])',
                },
                { prompt: '>>>', code: 'res' },
              ]}
              output={
                "Result(bfgs: converged after 38 iterations, x=array([1., 1.]), fun=4.669162237160902e-24, message='‖∇f‖∞ = 1.31e-11 ≤ gtol')"
              }
            />
            <p className={styles.prose}>
              Every iterate is on the record: <code>res.trace[1]</code> is the first BFGS step, with
              the line search's trials, the pair <code>s</code>, <code>y</code>, the curvature and
              the updated inverse Hessian <code>H</code> in its <code>info</code>.
            </p>
            <CodeBlock
              label="Python"
              lines={[
                { prompt: '>>>', code: 'step = res.trace[1]' },
                { prompt: '>>>', code: 'step.k, step.fun, step.step_size, step.info["curvature"]' },
              ]}
              output="(1, 12.212633421552631, 0.0013502003117837852, 110.04249751120305)"
            />
          </section>

          <section aria-labelledby="call">
            <h2 id="call" className={styles.h2} tabIndex={-1}>
              The call
            </h2>
            <p className={styles.prose}>
              Every method is registered under an id with its family, parameters (each with a
              default and a range), rate, summary and sources. <code>run</code> rejects unknown
              parameters, so a typo fails loudly. A method accepts a problem from the library or a
              bare callable; missing derivatives fall back to finite differences with exact counts.
            </p>
            <div className={styles.apiGrid}>
              {[
                [
                  'numopt.run(id, problem, *, x0=…, bracket=…, seed=…, **params)',
                  'Run any method; returns a Result.',
                ],
                ['numopt.minimize(f, x0=…, method="bfgs")', 'Minimize a callable.'],
                [
                  'numopt.find_root(f, bracket=(a, b), method="brent")',
                  'A zero of a scalar function.',
                ],
                ['numopt.list_methods(family=None)', 'The registry, as MethodSpec records.'],
                [
                  'numopt.problems.get(id) · list_problems(kind)',
                  'Test problems with exact derivatives, x₀ and known solutions.',
                ],
              ].map(([call, what]) => (
                <div key={call} className={styles.api}>
                  <code>{call}</code>
                  <span>{what}</span>
                </div>
              ))}
            </div>
            <CodeBlock
              label="Python"
              lines={[
                {
                  prompt: '>>>',
                  code: 'r = numopt.find_root(lambda x: x**3 - 2, bracket=(0, 2), method="brent")',
                },
                { prompt: '>>>', code: 'r.x, r.n_iter' },
              ]}
              output="(1.2599210498949451, 6)"
            />
          </section>

          <section aria-labelledby="records">
            <h2 id="records" className={styles.h2} tabIndex={-1}>
              Result and Step
            </h2>
            <p className={styles.prose}>
              A run returns one <code>Result</code>; its <code>trace</code> holds one{' '}
              <code>Step</code> per iterate. The trace is the didactic record: the labs animate it,
              and the parity fixtures are made of it.
            </p>
            <h3 className={styles.h3}>Result</h3>
            <Fields rows={RESULT_FIELDS} caption="Result fields" />
            <h3 className={styles.h3}>Step</h3>
            <Fields rows={STEP_FIELDS} caption="Step fields" />
          </section>

          <section aria-labelledby="cli">
            <h2 id="cli" className={styles.h2} tabIndex={-1}>
              The command line
            </h2>
            <p className={styles.prose}>
              <code>numopt list</code> and <code>numopt problems</code> browse the catalog;{' '}
              <code>run</code> and <code>compare</code> print one line per method (✓ when the
              stopping test passed, ✗ otherwise, with the reason); <code>--trace</code> prints every
              iterate and <code>--json</code> the full record.
            </p>
            <CodeBlock
              label="Shell"
              lines={[
                {
                  prompt: '$',
                  code: 'numopt compare rosenbrock gradient_descent momentum bfgs pure_newton',
                },
              ]}
              output={`✗ gradient_descent         iters=5000   fev=5294   f=5.079444109e-07    x=[0.9992873273, 0.9985745139]   (reached max_iter=5000 (‖∇f(x)‖ = 0.00117 > gtol))
✓ momentum                 iters=3020   fev=3021   f=1.248244184e-12    x=[0.9999988836, 0.9999977628]   (‖∇f(x)‖ = 9.98e-07 ≤ gtol)
✓ bfgs                     iters=38     fev=56     f=4.669162237e-24    x=[1, 1]   (‖∇f‖∞ = 1.31e-11 ≤ gtol)
✓ pure_newton              iters=6      fev=7      f=3.432646188e-20    x=[1, 1]   (‖∇f‖∞ = 7.41e-09 ≤ gtol; ∇²f is positive definite (λ_min = 0.399): a strict local minimizer)`}
            />
            <div className={styles.cliGrid}>
              {[
                ['numopt list [--family F]', 'methods'],
                ['numopt problems [--kind K]', 'test problems'],
                [
                  'numopt run METHOD PROBLEM [--x0 …] [--bracket a b] [--set k=v] [--trace] [--json]',
                  'one run',
                ],
                ['numopt compare PROBLEM M1 M2 …', 'one line per method'],
                [
                  'numopt bench P1 P2 … --methods M … --budget B [--tau T …] [--plot PREFIX]',
                  'performance and data profiles (Dolan–Moré, Moré–Wild)',
                ],
                ['numopt export OUT_DIR', 'registry, problems and parity fixtures as JSON'],
              ].map(([cmd, what]) => (
                <div key={cmd} className={styles.cli}>
                  <code>{cmd}</code>
                  <span>{what}</span>
                </div>
              ))}
            </div>
          </section>

          <section aria-labelledby="parity">
            <h2 id="parity" className={styles.h2} tabIndex={-1}>
              How parity works
            </h2>
            <p className={styles.prose}>
              The portal runs no Python. Every method here is a TypeScript port with the same id,
              parameters, stopping test, trace and <code>info</code> keys, and a test holds it to
              the Python record.
            </p>
            <ol className={styles.pipeline}>
              <li>
                <span className={styles.pipeHead}>1 · Register</span>
                <span>
                  <code>@register(id=…, params=(ParamSpec…), references=…)</code> in{' '}
                  <code>src/numopt/&lt;family&gt;/</code>; each module lists its{' '}
                  <code>FIXTURE_CASES</code>.
                </span>
              </li>
              <li>
                <span className={styles.pipeHead}>2 · Export</span>
                <span>
                  <code>numopt export</code> writes the registry, the problems and{' '}
                  {`${int(cases)} fixture cases`} — full traces — as JSON.
                </span>
              </li>
              <li>
                <span className={styles.pipeHead}>3 · Port</span>
                <span>
                  <code>web/src/methods/&lt;pkg&gt;/&lt;module&gt;.ts</code> keeps the Python order
                  of floating-point operations where the first iterates depend on it.
                </span>
              </li>
              <li>
                <span className={styles.pipeHead}>4 · Replay</span>
                <span>
                  <code>npm test</code> runs every case: first ten iterates within 10⁻⁸, the
                  iteration count exactly, the final iterate within 10⁻⁶ (relative), the same
                  verdict.
                </span>
              </li>
              <li>
                <span className={styles.pipeHead}>5 · Show</span>
                <span>
                  Each <a href="#/methods">method page</a> replays its cases again in your browser
                  and shows the comparison.
                </span>
              </li>
            </ol>
            <p className={styles.prose}>
              Seeded methods draw only from <code>numopt.core.rng.Rng</code>, a Mulberry32 generator
              that is bit-identical in TypeScript (tested against 1,400 Python values), so they are
              checked like deterministic ones; libm differences in <code>log</code> and{' '}
              <code>cos</code> may move late iterates, so their check covers the first ten iterates
              and the final objective.
            </p>
          </section>

          <section aria-labelledby="more">
            <h2 id="more" className={styles.h2} tabIndex={-1}>
              Further reading
            </h2>
            <ul className={styles.links}>
              {[
                [
                  'Method contract',
                  `${ARCH}#method-contract-python`,
                  'signature, trace, convergence, counts, determinism',
                ],
                [
                  'Line-search API',
                  `${ARCH}#line-search-api-used-by-every-n-d-descent-method`,
                  'search(kind, f, grad, x, p, …) used by every n-D descent method',
                ],
                [
                  'Problem library',
                  `${ARCH}#problem-library`,
                  'exact derivatives, domains, known minima',
                ],
                [
                  'Testing rules',
                  `${ARCH}#testing-rules`,
                  'oracles, property tests, ruff and pyright',
                ],
                ['Web app', `${ARCH}#web-app`, 'how the portal is built'],
              ].map(([t, href, what]) => (
                <li key={t}>
                  <a href={href} target="_blank" rel="noreferrer">
                    {t} <Icon name="external" size={11} />
                  </a>
                  <span>{what}</span>
                </li>
              ))}
              <li>
                <a href="#/research">Research studies</a>
                <span>new methods tested against the package's baselines before promotion</span>
              </li>
            </ul>
          </section>
        </div>
      </div>
    </PageFrame>
  );
}
