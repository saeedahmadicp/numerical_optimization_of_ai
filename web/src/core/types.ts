/**
 * Core data types — a camelCase mirror of `numopt.core.types` / `numopt.core.registry`.
 *
 * The Python exporter writes snake_case JSON (`grad_norm`, `n_iter`, `A_ub`); `json.ts`
 * converts those payloads into these shapes. Ported methods construct them directly.
 */

/** A 1-D iterate is a number; an n-D iterate is a plain array (never a typed array, for JSON parity). */
export type Point = number | number[];
export type Vector = number[];
export type Matrix = number[][];

/** JSON-serializable geometry the visualizer reads (`bracket`, `direction`, `simplex`, ...). */
export type StepInfo = Record<string, unknown>;

/** One iteration of a method. `k = 0` is the starting state. */
export interface Step {
  k: number;
  x: Point;
  /** f(x) (minimization), residual (roots), or the current estimate (quadrature, ...). */
  fun: number | null;
  /** ‖∇f(x)‖₂ when the method has it. */
  gradNorm: number | null;
  /** Step length / trust radius / spacing that produced this iterate. */
  stepSize: number | null;
  info: StepInfo;
}

export interface Result {
  method: string;
  x: Point | unknown;
  fun: number | null;
  converged: boolean;
  message: string;
  nIter: number;
  nFev: number;
  nGev: number;
  nHev: number;
  trace: Step[];
  extra: Record<string, unknown>;
}

// ---------------------------------------------------------------------------------------
// Problems
// ---------------------------------------------------------------------------------------

export type ProblemKind =
  | 'roots'
  | 'systems'
  | 'scalar_min'
  | 'unconstrained'
  | 'least_squares'
  | 'stochastic'
  | 'constrained'
  | 'lp'
  | 'combinatorial'
  | 'linalg'
  | 'calculus'
  | 'data';

/** Metadata of an exported problem (`problems.json`); callables are not exported. */
export interface ProblemMeta {
  kind?: ProblemKind;
  id: string;
  name: string;
  latex: string;
  dim: number;
  /** 1-D: `[a, b]`; 2-D: `[[xmin, xmax], [ymin, ymax]]`. */
  domain: unknown[];
  x0: Point | null;
  bracket: [number, number] | null;
  minima: Point[];
  roots: Point[];
  constraints: { kind: 'ineq' | 'eq'; latex: string }[];
  exact: number | null;
  description: string;
  tags: string[];
}

export interface Constraint {
  kind: 'ineq' | 'eq';
  /** `fun(x) <= 0` for "ineq", `fun(x) == 0` for "eq". */
  fun: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  latex: string;
}

/**
 * A smooth problem with callables (TS port of `numopt.core.types.Problem`).
 *
 * Conventions match Python: for `dim == 1`, `f/grad/hess` take and return numbers; for `dim >= 2`,
 * `f(x) -> number`, `grad(x) -> Vector`, `hess(x) -> Matrix`. Systems: `f(x) -> Vector`, `jac`.
 */
export interface Problem<X = Point> {
  id: string;
  name: string;
  latex: string;
  dim: number;
  domain: unknown[];
  f: (x: X) => unknown;
  grad?: (x: X) => unknown;
  hess?: (x: X) => unknown;
  jac?: (x: X) => Matrix;
  residual?: (x: X) => Vector;
  x0?: X | null;
  bracket?: [number, number] | null;
  minima?: X[];
  roots?: X[];
  constraints?: Constraint[];
  exact?: number | null;
  description?: string;
  tags?: string[];
  extra?: Record<string, unknown>;
}

/** The common 2-D smooth minimization problem used by the unconstrained / global labs. */
export interface Problem2D extends Problem<Vector> {
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess?: (x: Vector) => Matrix;
  /** `[[xmin, xmax], [ymin, ymax]]` */
  domain: [[number, number], [number, number]];
}

export interface LinearProgram {
  id: string;
  name: string;
  c: Vector;
  aUb: Matrix | null;
  bUb: Vector | null;
  aEq: Matrix | null;
  bEq: Vector | null;
  sense: 'min' | 'max';
  integer: boolean[];
  optimum: Vector | null;
  optimalValue: number | null;
  description: string;
  domain: unknown[];
}

export interface LinearSystem {
  id: string;
  name: string;
  A: Matrix;
  b: Vector;
  solution: Vector | null;
  description: string;
  tags: string[];
}

export interface Dataset {
  id: string;
  name: string;
  x: Vector;
  y: Vector;
  fTrue?: (x: number) => number;
  latex: string;
  domain: [number, number] | null;
  description: string;
}

// ---------------------------------------------------------------------------------------
// Registry metadata
// ---------------------------------------------------------------------------------------

export const FAMILIES = [
  'roots',
  'systems',
  'scalar',
  'line_search',
  'unconstrained',
  'least_squares',
  'global',
  'stochastic',
  'constrained',
  'lp',
  'combinatorial',
  'linalg',
  'integration',
  'differentiation',
  'interpolation',
  'regression',
] as const;
export type Family = (typeof FAMILIES)[number];

export type ParamKind = 'float' | 'int' | 'bool' | 'choice' | 'vector';
export type ParamValue = number | boolean | string | number[];

export interface ParamSpec {
  name: string;
  default: ParamValue;
  kind: ParamKind;
  min: number | null;
  max: number | null;
  choices: string[];
  /** Ask the UI for a logarithmic slider (tolerances, learning rates). */
  log: boolean;
  help: string;
  /** TS-only display name ("Step size"); the UI falls back to the humanized `name`. */
  label?: string;
  /** TS-only TeX symbol shown before the label (`\\alpha`, `\\|\\nabla f\\| \\le`). */
  tex?: string;
  /**
   * TS-only display names of a choice's values (`{ strong_wolfe: 'Strong Wolfe', bb1: 'BB1
   * (long step)' }`); a value without one is shown with its underscores as spaces.
   */
  choiceLabels?: Readonly<Record<string, string>>;
}

export interface MethodSpec {
  id: string;
  family: Family;
  name: string;
  /**
   * TS-only short name for places where space forces it (a chart legend, a compact table),
   * always shown with `title={name}`; chips, cards and pages use `name`.
   */
  shortName?: string;
  params: ParamSpec[];
  needs: string[];
  order: string;
  summary: string;
  references: string[];
  deterministic: boolean;
  tags: string[];
}

export type Params = Record<string, ParamValue>;

/** Keyword arguments every method accepts besides its declared params. */
export interface RunOptions {
  x0?: Point;
  bracket?: [number, number];
  seed?: number;
}

/** A TS method: same contract as Python `fn(problem, *, x0|bracket|seed, **params) -> Result`. */
export type MethodFn<P = unknown> = (problem: P, options: RunOptions & Params) => Result;
