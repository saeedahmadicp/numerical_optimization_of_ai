/** Shapes of the build-time catalog (see web/vite/catalog.ts, which produces them). */

export interface CatalogCounts {
  /** Methods in the Python reference (registry.json). */
  methods: number;
  families: number;
  problems: number;
  research: number;
  byFamily: Record<string, number>;
  problemsByKind: Record<string, number>;
}

/** A ParamSpec as exported by Python (registry.json). */
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
  /** What the method needs from a problem (`f`, `grad`, `hess`, `bracket`, `data`, …). */
  needs: string[];
  params: CatalogParam[];
  /** Number of parity fixture cases. */
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
  /** One curated sentence with the result, or null while the write-up is in progress. */
  finding: string | null;
  status: string;
}

export interface CatalogIndex {
  methods: CatalogMethod[];
  problems: CatalogProblem[];
  research: CatalogResearch[];
}

/** One parity fixture case, reduced (web/vite/catalog.ts `readParityCases`). */
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
  /** The first min(10, n) iterates. */
  head: unknown[];
  steps: number;
}

export interface StudyHeading {
  depth: number;
  text: string;
  slug: string;
}

/** One research study rendered at build time (web/vite/research.ts). HTML is trusted repo content. */
export interface StudyDoc {
  id: string;
  title: string;
  lede: string;
  html: string;
  toc: StudyHeading[];
  figures: string[];
  missing: string[];
  words: number;
}

export interface StudySummary {
  id: string;
  title: string;
  tags: string;
  family: string;
  /** HTML (math typeset). */
  question: string;
  finding: string;
  promote: string;
  figure: string | null;
  figureAlt: string;
  words: number;
  figures: number;
}
