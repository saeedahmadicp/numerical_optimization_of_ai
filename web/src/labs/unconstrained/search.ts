/** The line-search rule behind a step, as the Line-search panel names and draws it. */

/**
 * The tests of numopt.line_search (C1_DEFAULT = 1e-4, C_GOLDSTEIN_DEFAULT = 0.25) and of the
 * Barzilai–Borwein GLL search (BB_GAMMA = 1e-4).
 */
export type SearchRule =
  | 'backtracking'
  | 'strong_wolfe'
  | 'weak_wolfe'
  | 'goldstein'
  | 'exact_quadratic'
  | 'gll'
  | 'fixed'
  | 'bb'
  | 'newton'
  | 'schedule'
  | 'fista';

export const SEARCH: Record<SearchRule, { name: string; c1: number | null }> = {
  backtracking: { name: 'Armijo backtracking', c1: 1e-4 },
  strong_wolfe: { name: 'strong Wolfe search', c1: 1e-4 },
  weak_wolfe: { name: 'weak Wolfe search', c1: 1e-4 },
  goldstein: { name: 'Goldstein search', c1: 0.25 },
  exact_quadratic: { name: 'exact quadratic step', c1: null },
  gll: { name: 'nonmonotone (GLL) backtracking', c1: 1e-4 },
  fixed: { name: 'fixed step', c1: null },
  bb: { name: 'Barzilai–Borwein step', c1: null },
  newton: { name: 'Newton step', c1: null },
  schedule: { name: 'step-size schedule', c1: null },
  // Beck–Teboulle's test f(p) ≤ f(y) − ‖∇f(y)‖²/(2L̄) is Armijo's with c₁ = ½ along −∇f(y).
  fista: { name: 'Beck–Teboulle backtracking', c1: 0.5 },
};

/**
 * The search a method ran, from its id and parameters: pure Newton and pure BB
 * (`nonmonotone = false`) take their step without a test.
 */
export function searchRuleOf(
  method: string,
  params: Readonly<Record<string, unknown>>,
): SearchRule {
  if (method === 'pure_newton') return 'newton';
  if (method === 'barzilai_borwein') return params.nonmonotone === false ? 'bb' : 'gll';
  if (method === 'silver_gd' || method === 'silver_gd_strongly_convex' || method === 'long_step_gd')
    return 'schedule';
  if (method === 'fista') return params.backtracking === false ? 'fixed' : 'fista';
  const r = String(params.line_search ?? params.step_rule ?? 'backtracking');
  return r in SEARCH ? (r as SearchRule) : 'backtracking';
}
