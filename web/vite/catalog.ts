/**
 * Build-time catalog: reads `src/generated/registry.json` + `problems.json` (the `numopt export`
 * output) and the research notes, and exposes two small virtual modules:
 *
 *   virtual:numopt/catalog  — counts only (a few hundred bytes; the home page imports it eagerly)
 *   virtual:numopt/index    — methods, problems and research notes for search, the Methods page,
 *                             the Research page and planned-lab pages (imported lazily)
 *
 * The raw JSON (≈ 350 KB) never reaches the bundle, and parity fixtures are never read here.
 * Missing files give an empty catalog, so the app builds before the Python side has been run.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';
import { ABOUT_ID, figurePath, readStudies } from './research';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, '..');
const REPO = resolve(WEB, '..');
const REGISTRY = join(WEB, 'src/generated/registry.json');
const PROBLEMS = join(WEB, 'src/generated/problems.json');
const FACTS = join(REPO, 'docs/brand/portal/facts.js');
const RESEARCH = join(REPO, 'research');

const FIXTURES = join(WEB, 'src/generated/fixtures');
const REPO_URL = 'https://github.com/saeedahmadicp/numerical_optimization_of_ai';

const CATALOG_ID = 'virtual:numopt/catalog';
const INDEX_ID = 'virtual:numopt/index';
/** `virtual:numopt/parity/<family>`: a compact record of every fixture case of one family. */
const PARITY_PREFIX = 'virtual:numopt/parity/';
/** `virtual:numopt/parity`: `loaders[family]()` → that family's parity module. */
const PARITY_ID = 'virtual:numopt/parity';
/** `virtual:numopt/research`: study summaries + a lazy loader per study. */
const RESEARCH_ID = 'virtual:numopt/research';
/** `virtual:numopt/research/<id>`: one study rendered to HTML (KaTeX typeset at build time). */
const STUDY_PREFIX = 'virtual:numopt/research/';

export interface CatalogParam {
  name: string;
  default: unknown;
  kind: string;
  min: number | null;
  max: number | null;
  choices: unknown[];
  log: boolean;
  help: string;
}
export interface CatalogMethod {
  id: string;
  family: string;
  name: string;
  order: string;
  summary: string;
  references: string[];
  deterministic: boolean;
  tags: string[];
  needs: string[];
  params: CatalogParam[];
  /** Parity fixture cases for this method (src/generated/fixtures). */
  cases: number;
}
export interface CatalogProblem {
  id: string;
  kind: string;
  name: string;
  latex: string;
  dim: number;
  description: string;
  tags: string[];
}
export interface CatalogResearch {
  id: string;
  title: string;
  finding: string | null;
  status: string;
}

function readJson(path: string): unknown {
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return null;
  }
}

const str = (v: unknown) => (typeof v === 'string' ? v : '');
const strs = (v: unknown) =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];

const num = (v: unknown) => (typeof v === 'number' ? v : null);

export function readMethods(path = REGISTRY): CatalogMethod[] {
  const raw = readJson(path);
  if (!Array.isArray(raw)) return [];
  const cases = countBy(readParityCases(), (c) => c.method);
  return raw.map((m: Record<string, unknown>) => ({
    id: str(m.id),
    family: str(m.family),
    name: str(m.name),
    order: str(m.order),
    summary: str(m.summary),
    references: strs(m.references),
    deterministic: m.deterministic !== false,
    tags: strs(m.tags),
    needs: strs(m.needs),
    params: (Array.isArray(m.params) ? m.params : []).map((p: Record<string, unknown>) => ({
      name: str(p.name),
      default: p.default ?? null,
      kind: str(p.kind),
      min: num(p.min),
      max: num(p.max),
      choices: Array.isArray(p.choices) ? p.choices : [],
      log: p.log === true,
      help: str(p.help),
    })),
    cases: cases[str(m.id)] ?? 0,
  }));
}

/** One fixture case, reduced to what the method page needs to replay it in the browser. */
export interface ParityCase {
  family: string;
  method: string;
  problem: string;
  params: Record<string, unknown>;
  nIter: number;
  converged: boolean;
  fun: unknown;
  x: unknown;
  message: string;
  /** The first min(10, n) iterates (the harness compares these within 1e-8). */
  head: unknown[];
  /** Trace length. */
  steps: number;
}

/**
 * The parity fixtures, reduced: each case keeps its inputs, the final record and the first ten
 * iterates (≈ 120 KB for all 226 cases, against ≈ 10 MB of fixtures). The fixtures themselves are
 * never bundled; this record lets a method page rerun the parity check live.
 */
export function readParityCases(dir = FIXTURES): ParityCase[] {
  if (!existsSync(dir)) return [];
  const out: ParityCase[] = [];
  for (const f of readdirSync(dir)
    .filter((n) => n.endsWith('.json'))
    .sort()) {
    const raw = readJson(join(dir, f));
    if (!Array.isArray(raw)) continue;
    for (const c of raw as Record<string, unknown>[]) {
      const r = (c.result ?? {}) as Record<string, unknown>;
      const trace = Array.isArray(r.trace) ? (r.trace as Record<string, unknown>[]) : [];
      out.push({
        family: f.replace(/\.json$/, ''),
        method: str(c.method),
        problem: str(c.problem),
        params: (c.params ?? {}) as Record<string, unknown>,
        nIter: typeof r.n_iter === 'number' ? r.n_iter : 0,
        converged: r.converged === true,
        fun: r.fun ?? null,
        x: r.x ?? null,
        message: str(r.message),
        head: trace.slice(0, 10).map((t) => t.x ?? null),
        steps: trace.length,
      });
    }
  }
  return out;
}

export function readProblems(path = PROBLEMS): CatalogProblem[] {
  const raw = readJson(path);
  if (!Array.isArray(raw)) return [];
  return raw.map((p: Record<string, unknown>) => ({
    id: str(p.id),
    kind: str(p.kind),
    name: str(p.name),
    latex: str(p.latex),
    dim: typeof p.dim === 'number' ? p.dim : 0,
    description: str(p.description),
    tags: strs(p.tags),
  }));
}

/** Research notes: the curated list in facts.js when present, else the folders' README titles. */
export function readResearch(): CatalogResearch[] {
  if (existsSync(FACTS)) {
    const text = readFileSync(FACTS, 'utf8');
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    try {
      const facts = JSON.parse(text.slice(start, end + 1)) as { research?: CatalogResearch[] };
      if (Array.isArray(facts.research)) return facts.research;
    } catch {
      /* fall through to the folder scan */
    }
  }
  if (!existsSync(RESEARCH)) return [];
  return readdirSync(RESEARCH, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !/^[._]/.test(d.name))
    .map((d) => {
      const readme = join(RESEARCH, d.name, 'README.md');
      const title = existsSync(readme)
        ? (/^# (.+)$/m.exec(readFileSync(readme, 'utf8'))?.[1] ?? d.name)
        : d.name.replace(/-/g, ' ');
      return {
        id: d.name,
        title,
        finding: null,
        status: existsSync(readme) ? 'write-up' : 'prototype',
      };
    })
    .sort((a, b) => a.id.localeCompare(b.id));
}

export function countBy<T>(xs: readonly T[], key: (x: T) => string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const x of xs) out[key(x)] = (out[key(x)] ?? 0) + 1;
  return out;
}

export function catalogPlugin(): Plugin {
  const resolved = (id: string) => `\0${id}`;
  let build = false;
  let research: ReturnType<typeof readStudies> | null = null;
  const studies = () => (research ??= readStudies(RESEARCH, REPO_URL));
  const emitted = new Set<string>();
  return {
    name: 'numopt-catalog',
    configResolved(config) {
      build = config.command === 'build';
    },
    configureServer(server) {
      // Dev: serve research figures at the same relative URL the build emits them to.
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? '';
        if (!url.startsWith('/research/')) return next();
        const file = figurePath(RESEARCH, url);
        if (!file) return next();
        res.setHeader('Content-Type', file.endsWith('.svg') ? 'image/svg+xml' : 'image/png');
        res.setHeader('Cache-Control', 'no-cache');
        res.end(readFileSync(file));
      });
      server.watcher.add(join(RESEARCH, '*/README.md'));
      server.watcher.on('change', (f) => {
        if (f.startsWith(RESEARCH) && f.endsWith('README.md')) research = null;
      });
    },
    resolveId(id) {
      if (
        id === CATALOG_ID ||
        id === INDEX_ID ||
        id === RESEARCH_ID ||
        id === PARITY_ID ||
        id.startsWith(PARITY_PREFIX) ||
        id.startsWith(STUDY_PREFIX)
      )
        return resolved(id);
      return undefined;
    },
    load(id) {
      if (id.startsWith(resolved(PARITY_PREFIX))) {
        const family = id.slice(resolved(PARITY_PREFIX).length);
        const cases = readParityCases().filter((c) => c.family === family);
        if (existsSync(FIXTURES)) this.addWatchFile(join(FIXTURES, `${family}.json`));
        return `export default ${JSON.stringify(cases)};`;
      }
      if (id === resolved(PARITY_ID)) {
        const families = existsSync(FIXTURES)
          ? readdirSync(FIXTURES)
              .filter((n) => n.endsWith('.json'))
              .map((n) => n.replace(/\.json$/, ''))
              .sort()
          : [];
        const loaders = families
          .map((f) => `${JSON.stringify(f)}: () => import(${JSON.stringify(PARITY_PREFIX + f)})`)
          .join(',\n');
        return `export const loaders = {${loaders}};`;
      }
      if (id === resolved(RESEARCH_ID)) {
        const r = studies();
        for (const s of r.studies) this.addWatchFile(join(RESEARCH, s, 'README.md'));
        this.addWatchFile(join(RESEARCH, 'README.md'));
        const loaders = [...r.studies, ABOUT_ID]
          .map((s) => `${JSON.stringify(s)}: () => import(${JSON.stringify(STUDY_PREFIX + s)})`)
          .join(',\n');
        const about = r.load(ABOUT_ID);
        return [
          `export const studies = ${JSON.stringify(r.summaries())};`,
          `export const about = ${JSON.stringify({ title: about.title, lede: about.lede })};`,
          `export const loaders = {${loaders}};`,
        ].join('\n');
      }
      if (id.startsWith(resolved(STUDY_PREFIX))) {
        const sid = id.slice(resolved(STUDY_PREFIX).length);
        const r = studies();
        if (sid !== ABOUT_ID && !r.studies.includes(sid)) return 'export default null;';
        const doc = r.load(sid);
        if (build) {
          for (const f of doc.figures) {
            const url = sid === ABOUT_ID ? `research/${f}` : `research/${sid}/${f}`;
            const abs = figurePath(RESEARCH, url);
            if (!abs || emitted.has(url)) continue;
            emitted.add(url);
            this.emitFile({ type: 'asset', fileName: url, source: readFileSync(abs) });
          }
        }
        return `export default ${JSON.stringify(doc)};`;
      }
      if (id !== resolved(CATALOG_ID) && id !== resolved(INDEX_ID)) return undefined;
      for (const f of [REGISTRY, PROBLEMS, FACTS]) if (existsSync(f)) this.addWatchFile(f);
      const methods = readMethods();
      const problems = readProblems();
      const research = readResearch();
      if (id === resolved(CATALOG_ID)) {
        const counts = {
          methods: methods.length,
          families: Object.keys(countBy(methods, (m) => m.family)).length,
          problems: problems.length,
          research: research.length,
          byFamily: countBy(methods, (m) => m.family),
          problemsByKind: countBy(problems, (p) => p.kind),
        };
        return `export default ${JSON.stringify(counts)};`;
      }
      return [
        `export const methods = ${JSON.stringify(methods)};`,
        `export const problems = ${JSON.stringify(problems)};`,
        `export const research = ${JSON.stringify(research)};`,
      ].join('\n');
    },
  };
}

/** `__SITE_URL__` in index.html → the deployed origin + base (absolute og:image), or ''. */
export function siteUrlPlugin(siteUrl = process.env.SITE_URL ?? ''): Plugin {
  const base = siteUrl && !siteUrl.endsWith('/') ? `${siteUrl}/` : siteUrl;
  let warned = false;
  return {
    name: 'numopt-site-url',
    configResolved(config) {
      // Crawlers need an absolute og:image / twitter:image URL: say so in a production build.
      if (config.command === 'build' && !base && !warned) {
        warned = true;
        config.logger.warn(
          '\n[numopt] SITE_URL is not set: og:image and twitter:image stay relative, and link ' +
            'previews will have no image. Build the deploy with SITE_URL=https://<host>/<base>/.',
        );
      }
    },
    transformIndexHtml(html) {
      return html.replaceAll('__SITE_URL__', base);
    },
  };
}
