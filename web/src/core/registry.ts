/**
 * The TS method registry — mirror of `numopt.core.registry`.
 *
 * A ported module calls `registerMethod(spec, fn)` at import time with the SAME id, family and
 * params as its Python counterpart. Labs and the parity test look methods up by id.
 *
 *   registerMethod(
 *     { id: 'bfgs', family: 'unconstrained', name: 'BFGS', params: [p.float('gtol', 1e-8, ...)], ... },
 *     bfgs,
 *   );
 */
import {
  FAMILIES,
  type Family,
  type MethodFn,
  type MethodSpec,
  type ParamSpec,
  type ParamValue,
} from './types';

/**
 * Teaching material for the MethodCard (TS-only; Python keeps this in docstrings).
 * `quantities` name the Step fields the card shows live next to the update rule.
 */
export interface MethodDoc {
  /** Display-mode LaTeX of the update rule, e.g. `x_{k+1} = x_k - \\alpha \\nabla f(x_k)`. */
  rule: string;
  /** One or two plain sentences: what the method does, in pictures. */
  intuition: string;
  /** Convergence order in words ("linear", "quadratic", "superlinear"). Defaults to `spec.order`. */
  order?: string;
  pros?: string[];
  cons?: string[];
  /** Live quantities: `{ tex: '\\alpha_k', key: 'stepSize' }` reads `step.stepSize`; `info.<key>` reads `step.info[key]`. */
  quantities?: { tex: string; key: string; label?: string }[];
}

export interface RegisteredMethod<P = unknown> {
  spec: MethodSpec;
  fn: MethodFn<P>;
  doc?: MethodDoc;
}

const REGISTRY = new Map<string, RegisteredMethod>();

/** Spec fields that may be omitted at registration (they get Python's defaults). */
export type MethodSpecInput = Pick<MethodSpec, 'id' | 'family' | 'name'> &
  Partial<Omit<MethodSpec, 'id' | 'family' | 'name'>>;

export function registerMethod<P>(
  input: MethodSpecInput,
  fn: MethodFn<P>,
  doc?: MethodDoc,
): RegisteredMethod<P> {
  if (!(FAMILIES as readonly string[]).includes(input.family)) {
    throw new Error(`unknown family ${input.family}; expected one of ${FAMILIES.join(', ')}`);
  }
  const existing = REGISTRY.get(input.id);
  if (existing && existing.fn !== (fn as MethodFn)) {
    throw new Error(`method id ${input.id} is already registered`);
  }
  const spec: MethodSpec = {
    params: [],
    needs: [],
    order: '',
    summary: '',
    references: [],
    deterministic: true,
    tags: [],
    ...input,
  };
  const entry: RegisteredMethod = { spec, fn: fn as MethodFn, doc };
  REGISTRY.set(spec.id, entry);
  return entry as RegisteredMethod<P>;
}

export function getMethod<P = unknown>(id: string): RegisteredMethod<P> {
  const m = REGISTRY.get(id);
  if (!m) throw new Error(`unknown method ${id}`);
  return m as RegisteredMethod<P>;
}

export function hasMethod(id: string): boolean {
  return REGISTRY.has(id);
}

/** Registered methods sorted like Python: by family order, then id. */
export function listMethods(family?: Family): RegisteredMethod[] {
  return [...REGISTRY.values()]
    .filter((m) => family === undefined || m.spec.family === family)
    .sort(
      (a, b) =>
        FAMILIES.indexOf(a.spec.family) - FAMILIES.indexOf(b.spec.family) ||
        a.spec.id.localeCompare(b.spec.id),
    );
}

/** Test-only: forget every registration. */
export function clearRegistry(): void {
  REGISTRY.clear();
}

export function defaults(spec: Pick<MethodSpec, 'params'>): Record<string, ParamValue> {
  return Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
}

/** Run a registered method by id, rejecting unknown parameters like Python's `run()`. */
export function runMethod<P>(id: string, problem: P, params: Record<string, unknown> = {}) {
  const { spec, fn } = getMethod<P>(id);
  const allowed = new Set([...spec.params.map((p) => p.name), 'x0', 'bracket', 'seed']);
  const unknown = Object.keys(params).filter((k) => !allowed.has(k));
  if (unknown.length)
    throw new TypeError(`${id}: unknown parameter(s) ${unknown.sort().join(', ')}`);
  return fn(problem, { ...defaults(spec), ...(params as Record<string, ParamValue>) });
}

/** ParamSpec builders with Python's defaults, so ports read like the `@register` call. */
/**
 * Words a choice id spells in lower case that are names or acronyms: the fallback label keeps
 * them as they are written (`strong_wolfe` → "Strong Wolfe", `qr` → "QR").
 */
const CHOICE_WORDS: Readonly<Record<string, string>> = {
  armijo: 'Armijo',
  bfgs: 'BFGS',
  bland: 'Bland',
  cauchy: 'Cauchy',
  chebyshev: 'Chebyshev',
  dantzig: 'Dantzig',
  dfp: 'DFP',
  fletcher: 'Fletcher',
  gauss: 'Gauss',
  goldstein: 'Goldstein',
  hestenes: 'Hestenes',
  irls: 'IRLS',
  lbfgs: 'L-BFGS',
  lu: 'LU',
  newton: 'Newton',
  nesterov: 'Nesterov',
  polyak: 'Polyak',
  qr: 'QR',
  reeves: 'Reeves',
  ribiere: 'Ribière',
  seidel: 'Seidel',
  sr1: 'SR1',
  stiefel: 'Stiefel',
  svd: 'SVD',
  wolfe: 'Wolfe',
};

/**
 * A choice value's display name: `spec.choiceLabels`, else the id in sentence case with its
 * names and acronyms kept (`strong_wolfe` → "Strong Wolfe", `normal_equations` → "Normal
 * equations", `qr` → "QR").
 */
export const choiceLabel = (spec: Pick<ParamSpec, 'choiceLabels'>, c: string): string => {
  const own = spec.choiceLabels?.[c];
  if (own !== undefined) return own;
  const words = c.split('_').filter(Boolean);
  const text = words.map((w) => CHOICE_WORDS[w.toLowerCase()] ?? w).join(' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
};

export const param = {
  float: (name: string, def: number, o: Partial<ParamSpec> = {}): ParamSpec => ({
    name,
    default: def,
    kind: 'float',
    min: null,
    max: null,
    choices: [],
    log: false,
    help: '',
    ...o,
  }),
  int: (name: string, def: number, o: Partial<ParamSpec> = {}): ParamSpec => ({
    name,
    default: def,
    kind: 'int',
    min: null,
    max: null,
    choices: [],
    log: false,
    help: '',
    ...o,
  }),
  bool: (name: string, def: boolean, o: Partial<ParamSpec> = {}): ParamSpec => ({
    name,
    default: def,
    kind: 'bool',
    min: null,
    max: null,
    choices: [],
    log: false,
    help: '',
    ...o,
  }),
  choice: (
    name: string,
    def: string,
    choices: string[],
    o: Partial<ParamSpec> = {},
  ): ParamSpec => ({
    name,
    default: def,
    kind: 'choice',
    min: null,
    max: null,
    choices,
    log: false,
    help: '',
    ...o,
  }),
};
