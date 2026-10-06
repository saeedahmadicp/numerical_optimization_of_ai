/**
 * The 42 methods of the `unconstrained` family as the lab shows them: their picker group, the
 * kind of step geometry they draw, and the MethodCard material in the lab's notation (vectors
 * bold, coordinates italic, a short rate word for the badge). The ports' own `doc` supplies the
 * rest (intuition, strengths, weaknesses) wherever it has it.
 */
import type { MethodDoc, RegisteredMethod } from '../../core/registry';

/** How a method's step is drawn on the landscape (geometry.ts). */
export type Kind =
  | 'line' //       gradient descent, Barzilai–Borwein: a ray and its line-search trials
  | 'coord' //      cyclic coordinate descent: one axis per iteration
  | 'heavy' //      heavy-ball momentum: βv + (−α∇f), tip to tail
  | 'nesterov' //   look-ahead point and the gradient taken there
  | 'adaptive' //   per-coordinate preconditioner: the step ellipse
  | 'newton' //     the quadratic model through 𝐱ₖ₋₁ and its center
  | 'qn' //         the quasi-Newton model ellipse (from Hₖ) vs the true Hessian's
  | 'cg' //         d = −∇f + βd_prev, tip to tail, and the level set the last search touched
  | 'tr' //         the trust region, the model's level set, Cauchy / Newton points, dogleg
  | 'nm' //         the simplex and its operation
  | 'powell' //     the sweep of line minimizations and the extrapolated point
  | 'hj' //         exploratory probes and the pattern move
  | 'compass' //    the compass poll
  | 'schedule' //   a step of a fixed schedule h_k/L along −∇f, and the classical 1/L
  | 'ogm' //        OGM: the gradient step to y_k, then the two momentum terms
  | 'fista' //      FISTA: momentum to y_k, the backtracked gradient step there, restarts
  | 'anderson' //   AA(m): the mixed history window, x̄ = Σcᵢxᵢ and the step
  | 'arc' //        ARC: the cubic model's level set and the ball ‖s‖ = λ/σ
  | 'regnewton'; // regularized Newton: the shifted model and the path λ ↦ −(B + λI)⁻¹∇f

export interface MethodGroup {
  id: string;
  label: string;
  methods: readonly string[];
}

/** Picker groups, in syllabus order. Every unconstrained method appears exactly once. */
export const GROUPS: readonly MethodGroup[] = [
  {
    id: 'gradient',
    label: 'Gradient methods',
    methods: ['gradient_descent', 'barzilai_borwein', 'coordinate_descent'],
  },
  {
    id: 'schedule',
    label: 'Certified step-size schedules',
    methods: ['silver_gd', 'silver_gd_strongly_convex', 'long_step_gd'],
  },
  {
    id: 'momentum',
    label: 'Momentum and acceleration',
    methods: ['momentum', 'nesterov', 'ogm', 'fista', 'anderson_gd'],
  },
  {
    id: 'adaptive',
    label: 'Adaptive (machine-learning optimizers)',
    methods: ['adagrad', 'rmsprop', 'adadelta', 'adam', 'adamw', 'adamax', 'nadam', 'amsgrad'],
  },
  { id: 'newton', label: 'Newton', methods: ['pure_newton', 'damped_newton', 'modified_newton'] },
  { id: 'qn', label: 'Quasi-Newton', methods: ['bfgs', 'dfp', 'sr1', 'broyden_class', 'lbfgs'] },
  {
    id: 'cg',
    label: 'Nonlinear conjugate gradient',
    methods: [
      'cg_fletcher_reeves',
      'cg_polak_ribiere',
      'cg_hestenes_stiefel',
      'cg_dai_yuan',
      'cg_hager_zhang',
    ],
  },
  {
    id: 'tr',
    label: 'Trust region',
    methods: [
      'trust_region_cauchy',
      'trust_region_dogleg',
      'trust_region_steihaug',
      'trust_region_exact',
    ],
  },
  { id: 'regnewton', label: 'Regularized Newton', methods: ['arc', 'reg_newton'] },
  {
    id: 'df',
    label: 'Derivative-free',
    methods: ['nelder_mead', 'powell', 'hooke_jeeves', 'compass_search'],
  },
];

const GROUP_OF = new Map(GROUPS.flatMap((g) => g.methods.map((m) => [m, g] as const)));
export const groupOf = (id: string): MethodGroup | undefined => GROUP_OF.get(id);

const KINDS: Record<string, Kind> = {
  gradient_descent: 'line',
  barzilai_borwein: 'line',
  coordinate_descent: 'coord',
  momentum: 'heavy',
  nesterov: 'nesterov',
  adagrad: 'adaptive',
  rmsprop: 'adaptive',
  adadelta: 'adaptive',
  adam: 'adaptive',
  adamw: 'adaptive',
  adamax: 'adaptive',
  nadam: 'adaptive',
  amsgrad: 'adaptive',
  pure_newton: 'newton',
  damped_newton: 'newton',
  modified_newton: 'newton',
  bfgs: 'qn',
  dfp: 'qn',
  sr1: 'qn',
  broyden_class: 'qn',
  lbfgs: 'qn',
  cg_fletcher_reeves: 'cg',
  cg_polak_ribiere: 'cg',
  cg_hestenes_stiefel: 'cg',
  cg_dai_yuan: 'cg',
  cg_hager_zhang: 'cg',
  trust_region_cauchy: 'tr',
  trust_region_dogleg: 'tr',
  trust_region_steihaug: 'tr',
  trust_region_exact: 'tr',
  nelder_mead: 'nm',
  powell: 'powell',
  hooke_jeeves: 'hj',
  compass_search: 'compass',
  silver_gd: 'schedule',
  silver_gd_strongly_convex: 'schedule',
  long_step_gd: 'schedule',
  ogm: 'ogm',
  fista: 'fista',
  anderson_gd: 'anderson',
  arc: 'arc',
  reg_newton: 'regnewton',
};

export const kindOf = (id: string): Kind | undefined => KINDS[id];
export const KNOWN_METHODS: readonly string[] = Object.keys(KINDS);

/** Methods whose step comes from a line search along a direction (the φ(α) panel applies). */
export const RAY_KINDS: ReadonlySet<Kind> = new Set([
  'line',
  'coord',
  'newton',
  'qn',
  'cg',
  'schedule',
  'fista',
]);

/** Kinds whose stopping test is ‖∇f‖∞ ≤ gtol (the others test ‖∇f‖₂). */
export const INF_NORM_KINDS: ReadonlySet<Kind> = new Set([
  'newton',
  'qn',
  'cg',
  'tr',
  'arc',
  'regnewton',
]);

/** Methods that never evaluate ∇f (their ‖∇f‖ in the chart is evaluated by the lab). */
export const DERIVATIVE_FREE: ReadonlySet<Kind> = new Set(['nm', 'powell', 'hj', 'compass']);

/**
 * A proven local rate, annotated at the end of a converged curve (brand.md §9: only when the
 * method's theory guarantees it, here under its standing assumption ∇²f(𝐱⋆) ≻ 0).
 */
export const PROVEN_RATE: Readonly<Record<string, string>> = {
  pure_newton: 'quadratic',
  damped_newton: 'quadratic',
  modified_newton: 'quadratic',
  trust_region_dogleg: 'quadratic',
  trust_region_exact: 'quadratic',
  bfgs: 'superlinear',
  dfp: 'superlinear',
  sr1: 'superlinear',
  broyden_class: 'superlinear',
  trust_region_steihaug: 'superlinear',
  arc: 'superlinear',
  silver_gd_strongly_convex: 'linear',
};

// ── MethodCard material in the lab's notation ────────────────────────────────────────

const X = '\\mathbf{x}';
const G = (i: string) => `\\nabla f(${X}_{${i}})`;
const STEP = `${X}_k = ${X}_{k-1} + \\alpha_k\\,\\mathbf{p}_k`;
const QN_DIR = `\\mathbf{p}_k = -H_{k-1}\\,${G('k-1')}`;
const CG_DIR = `\\mathbf{d}_k = -${G('k')} + \\beta_k\\,\\mathbf{d}_{k-1}`;
const TR_MODEL = `\\min_{\\|\\mathbf{p}\\| \\le \\Delta_k} m_k(\\mathbf{p}) = f_k + \\nabla f_k^{\\top}\\mathbf{p} + \\tfrac12\\,\\mathbf{p}^{\\top}\\nabla^2 f_k\\,\\mathbf{p}`;
const ADAM_V = `\\hat{\\mathbf{v}}_k`;

type LocalDoc = Pick<MethodDoc, 'rule' | 'order'> & Partial<Omit<MethodDoc, 'rule' | 'order'>>;

const TR_PROS = ['Never takes a step the model does not trust', 'Uses negative curvature safely'];
const GD_L = `${X}_k = ${X}_{k-1} - \\frac{h_{k-1}}{L}\\,${G('k-1')}`;
const SCHEDULE_CONS = [
  'Needs a global Lipschitz constant L of ∇f; with L too small the long steps diverge',
  'f is not monotone: a long step can raise it far above f(𝐱₀) before the short steps recover',
];

export const LAB_DOCS: Readonly<Record<string, LocalDoc>> = {
  gradient_descent: {
    rule: `${X}_k = ${X}_{k-1} - \\alpha_k\\,${G('k-1')}`,
    order: 'linear',
  },
  barzilai_borwein: {
    rule: `\\alpha_k = \\frac{\\mathbf{s}^{\\top}\\mathbf{s}}{\\mathbf{s}^{\\top}\\mathbf{y}}\\ \\text{(BB1)},\\quad ${X}_k = ${X}_{k-1} - \\alpha_k\\,${G('k-1')},\\quad \\mathbf{s} = ${X}_{k-1} - ${X}_{k-2},\\ \\mathbf{y} = ${G('k-1')} - ${G('k-2')}`,
    order: 'R-linear',
  },
  coordinate_descent: {
    rule: `${X}_k = ${X}_{k-1} - \\alpha_k\\,\\frac{\\partial_i f(${X}_{k-1})}{\\partial_{ii}^2 f(${X}_{k-1})}\\,\\mathbf{e}_i,\\quad i = (k-1) \\bmod n`,
    order: 'linear',
  },
  momentum: {
    rule: `\\mathbf{v}_k = \\beta\\,\\mathbf{v}_{k-1} - \\alpha\\,${G('k-1')},\\quad ${X}_k = ${X}_{k-1} + \\mathbf{v}_k`,
    order: 'linear',
  },
  nesterov: {
    rule: `\\mathbf{v}_k = \\mu\\,\\mathbf{v}_{k-1} - \\alpha\\,\\nabla f(${X}_{k-1} + \\mu\\,\\mathbf{v}_{k-1}),\\quad ${X}_k = ${X}_{k-1} + \\mathbf{v}_k`,
    order: 'linear',
  },
  adagrad: {
    rule: `\\mathbf{G}_k = \\mathbf{G}_{k-1} + \\mathbf{g}^{2},\\quad ${X}_k = ${X}_{k-1} - \\frac{\\alpha}{\\sqrt{\\mathbf{G}_k} + \\varepsilon} \\odot \\mathbf{g},\\quad \\mathbf{g} = ${G('k-1')}`,
    order: 'sublinear',
  },
  rmsprop: {
    rule: `\\mathbf{E}_k = \\rho\\,\\mathbf{E}_{k-1} + (1-\\rho)\\,\\mathbf{g}^{2},\\quad ${X}_k = ${X}_{k-1} - \\frac{\\alpha}{\\sqrt{\\mathbf{E}_k} + \\varepsilon} \\odot \\mathbf{g}`,
    order: 'no guarantee',
  },
  adadelta: {
    rule: `${X}_k = ${X}_{k-1} - \\frac{\\operatorname{RMS}[\\Delta ${X}]_{k-1}}{\\operatorname{RMS}[\\mathbf{g}]_k} \\odot \\mathbf{g},\\quad \\mathbf{g} = ${G('k-1')}`,
    order: 'no guarantee',
  },
  adam: {
    rule: `${X}_k = ${X}_{k-1} - \\frac{\\alpha}{\\sqrt{${ADAM_V}} + \\varepsilon} \\odot \\hat{\\mathbf{m}}_k,\\quad \\hat{\\mathbf{m}}_k = \\frac{\\mathbf{m}_k}{1-\\beta_1^{k}},\\ ${ADAM_V} = \\frac{\\mathbf{v}_k}{1-\\beta_2^{k}}`,
    order: 'no guarantee',
  },
  adamw: {
    rule: `${X}_k = ${X}_{k-1} - \\frac{\\alpha}{\\sqrt{${ADAM_V}} + \\varepsilon} \\odot \\hat{\\mathbf{m}}_k - \\lambda\\,${X}_{k-1}`,
    order: 'no guarantee',
  },
  adamax: {
    rule: `\\mathbf{u}_k = \\max(\\beta_2\\mathbf{u}_{k-1}, |\\mathbf{g}|),\\quad ${X}_k = ${X}_{k-1} - \\frac{\\alpha}{(1-\\beta_1^{k})\\,\\mathbf{u}_k} \\odot \\mathbf{m}_k`,
    order: 'no guarantee',
  },
  nadam: {
    rule: `${X}_k = ${X}_{k-1} - \\frac{\\alpha}{\\sqrt{${ADAM_V}} + \\varepsilon} \\odot \\bar{\\mathbf{m}}_k,\\quad \\bar{\\mathbf{m}}_k = \\beta_1 \\hat{\\mathbf{m}}_k + (1-\\beta_1)\\,\\hat{\\mathbf{g}}`,
    order: 'no guarantee',
  },
  amsgrad: {
    rule: `\\hat{\\mathbf{v}}_k = \\max(\\hat{\\mathbf{v}}_{k-1}, \\mathbf{v}_k),\\quad ${X}_k = ${X}_{k-1} - \\frac{\\alpha}{\\sqrt{\\hat{\\mathbf{v}}_k}} \\odot \\mathbf{m}_k`,
    order: 'linear (local)',
  },
  pure_newton: {
    rule: `\\nabla^2 f(${X}_{k-1})\\,\\mathbf{p}_k = -${G('k-1')},\\quad ${X}_k = ${X}_{k-1} + \\mathbf{p}_k`,
    order: 'quadratic',
  },
  damped_newton: {
    rule: `\\nabla^2 f(${X}_{k-1})\\,\\mathbf{p}_k = -${G('k-1')},\\quad ${STEP}`,
    order: 'quadratic',
  },
  modified_newton: {
    rule: `(\\nabla^2 f(${X}_{k-1}) + \\tau_k I)\\,\\mathbf{p}_k = -${G('k-1')},\\quad ${STEP}`,
    order: 'quadratic',
  },
  bfgs: {
    rule: `${QN_DIR},\\quad H_k = (I - \\rho\\,\\mathbf{s}\\mathbf{y}^{\\top})H_{k-1}(I - \\rho\\,\\mathbf{y}\\mathbf{s}^{\\top}) + \\rho\\,\\mathbf{s}\\mathbf{s}^{\\top},\\quad \\rho = \\tfrac{1}{\\mathbf{y}^{\\top}\\mathbf{s}}`,
    order: 'superlinear',
    intuition:
      'Learn the curvature from how the gradient changes along each step. A rank-two correction ' +
      'makes the inverse-Hessian estimate map the gradient change onto the step (the secant ' +
      'equation) and keeps it positive definite.',
  },
  dfp: {
    rule: `${QN_DIR},\\quad H_k = H_{k-1} - \\frac{H_{k-1}\\mathbf{y}\\mathbf{y}^{\\top}H_{k-1}}{\\mathbf{y}^{\\top}H_{k-1}\\mathbf{y}} + \\frac{\\mathbf{s}\\mathbf{s}^{\\top}}{\\mathbf{y}^{\\top}\\mathbf{s}}`,
    order: 'superlinear',
  },
  sr1: {
    rule: `${QN_DIR},\\quad H_k = H_{k-1} + \\frac{\\mathbf{w}\\mathbf{w}^{\\top}}{\\mathbf{w}^{\\top}\\mathbf{y}},\\quad \\mathbf{w} = \\mathbf{s} - H_{k-1}\\mathbf{y}`,
    order: 'superlinear',
  },
  broyden_class: {
    rule: `${QN_DIR},\\quad B_k = (1-\\phi)\\,B_k^{\\mathrm{BFGS}} + \\phi\\,B_k^{\\mathrm{DFP}},\\ \\phi \\in [0, 1]`,
    order: 'superlinear',
  },
  lbfgs: {
    rule: `${QN_DIR},\\quad H_{k-1} = \\text{BFGS}^{m}\\big(\\gamma I;\\ (\\mathbf{s}_i, \\mathbf{y}_i)_{i > k-1-m}\\big)`,
    order: 'linear',
  },
  cg_fletcher_reeves: {
    rule: `${CG_DIR},\\quad \\beta_k = \\frac{\\|${G('k')}\\|^2}{\\|${G('k-1')}\\|^2}`,
    order: 'linear',
    pros: [
      'Ends in n steps on a quadratic with exact line searches',
      'Stores two vectors, no matrix',
    ],
    cons: [
      'After one poor step it can take many tiny ones (no automatic restart)',
      'Needs c₂ < ½ in the Wolfe search for guaranteed descent',
    ],
  },
  cg_polak_ribiere: {
    rule: `${CG_DIR},\\quad \\beta_k = \\max\\Big(\\frac{${G('k')}^{\\top}\\mathbf{y}}{\\|${G('k-1')}\\|^2},\\ 0\\Big)`,
    order: 'linear',
    pros: [
      'Restarts itself when progress stalls (β is truncated at 0)',
      'Stores two vectors, no matrix',
    ],
    cons: ['Descent needs a Wolfe search; without the truncation it can cycle (Powell 1984)'],
  },
  cg_hestenes_stiefel: {
    rule: `${CG_DIR},\\quad \\beta_k = \\frac{${G('k')}^{\\top}\\mathbf{y}}{\\mathbf{d}_{k-1}^{\\top}\\mathbf{y}}`,
    order: 'linear',
    pros: ['Keeps 𝐝ₖᵀ𝐲 = 0, the conjugacy condition, by construction'],
    cons: ['Like Polak–Ribière, no descent guarantee without restarts'],
  },
  cg_dai_yuan: {
    rule: `${CG_DIR},\\quad \\beta_k = \\frac{\\|${G('k')}\\|^2}{\\mathbf{d}_{k-1}^{\\top}\\mathbf{y}}`,
    order: 'linear',
    pros: ['Descent with any Wolfe line search (Dai & Yuan 1999)'],
    cons: ['Slow in practice: its steps shrink like Fletcher–Reeves’'],
  },
  cg_hager_zhang: {
    intuition:
      'Hestenes–Stiefel with a correction term that keeps every direction a sufficient descent ' +
      'direction, and a lower bound on β that stops it from going too negative.',
    rule: `${CG_DIR},\\quad \\beta_k = \\max\\Big(\\Big(\\mathbf{y} - 2\\mathbf{d}_{k-1}\\tfrac{\\|\\mathbf{y}\\|^2}{\\mathbf{d}_{k-1}^{\\top}\\mathbf{y}}\\Big)^{\\!\\top}\\frac{${G('k')}}{\\mathbf{d}_{k-1}^{\\top}\\mathbf{y}},\\ \\eta_k\\Big)`,
    order: 'linear',
    pros: ['Sufficient descent whatever the line search (Hager & Zhang 2005)'],
    cons: ['More work per β; the lower bound ηₖ has a tuned constant'],
  },
  trust_region_cauchy: {
    rule: `${TR_MODEL},\\quad \\mathbf{p}^{C} = -\\tau\\,\\frac{\\Delta_k}{\\|\\nabla f_k\\|}\\,\\nabla f_k`,
    order: 'linear',
    pros: ['Global convergence with only the Cauchy decrease', 'Cheap: one Hessian–vector product'],
    cons: [
      'Steepest descent in disguise: crawls along curved valleys',
      'Ignores the curvature the model has',
    ],
  },
  trust_region_dogleg: {
    rule: `${TR_MODEL},\\quad \\mathbf{p} \\in [\\mathbf{0}, \\mathbf{p}^{U}] \\cup [\\mathbf{p}^{U}, \\mathbf{p}^{B}],\\ \\|\\mathbf{p}\\| \\le \\Delta_k`,
    order: 'quadratic',
    pros: [...TR_PROS.slice(0, 1), 'Becomes Newton’s method once Δ contains 𝐩ᴮ'],
    cons: ['Needs ∇²f ≻ 0: falls back to the Cauchy point where it is indefinite'],
  },
  trust_region_steihaug: {
    rule: `${TR_MODEL},\\quad \\text{CG on } \\nabla^2 f_k\\,\\mathbf{p} = -\\nabla f_k \\text{ from } \\mathbf{z}_0 = \\mathbf{0}`,
    order: 'superlinear',
    pros: TR_PROS,
    cons: ['Stops at the first boundary hit, not at the subproblem’s minimizer'],
  },
  trust_region_exact: {
    rule: `(\\nabla^2 f_k + \\lambda I)\\,\\mathbf{p} = -\\nabla f_k,\\quad \\lambda \\ge 0,\\quad \\lambda\\,(\\Delta_k - \\|\\mathbf{p}\\|) = 0`,
    order: 'quadratic',
    pros: [...TR_PROS, 'Solves the subproblem globally, hard case included'],
    cons: ['An eigenvalue problem per step: too costly for large n'],
  },
  silver_gd: {
    rule: `${GD_L},\\quad h_t = 1 + \\rho^{\\nu(t+1) - 1},\\ \\rho = 1 + \\sqrt{2}`,
    order: '$O(k^{-1.27})$',
    intuition:
      'Gradient descent with a fixed fractal schedule: short steps √2, 2, √2 and, at every power ' +
      'of two, a longer step 1 + ρʲ. The long steps overshoot on purpose; the short ones repair ' +
      'the damage, and the net rate beats every constant step.',
    pros: [
      'Provably faster than any constant step on convex $f$: $f - f^\\star \\le r_j L\\|\\mathbf{x}_0 - \\mathbf{x}^\\star\\|^2$ at $k = 2^j - 1$',
      'No line search, no momentum: one gradient per step',
    ],
    cons: [
      ...SCHEDULE_CONS,
      'Ignores strong convexity: the rate stays polynomial even where it could be linear',
    ],
  },
  silver_gd_strongly_convex: {
    rule: `${GD_L},\\quad h = \\big[\\tilde h^{(n/2)},\\ a_n,\\ \\tilde h^{(n/2)},\\ b_n\\big] \\text{ repeated}`,
    order: 'linear',
    intuition:
      'The silver schedule tuned to κ = L/μ: a block of n steps, each between 1/L and ' +
      '(κ + 1)/(2L), that contracts ‖𝐱 − 𝐱⋆‖² by a certified factor τₙ. The block repeats.',
    pros: [
      'Certified linear rate: $O(\\kappa^{0.79}\\log\\frac1\\varepsilon)$ steps instead of $O(\\kappa\\log\\frac1\\varepsilon)$',
      'Every step stays below 1/μ, so no step can blow up',
    ],
    cons: ['Needs both L and μ; with a wrong μ the certificate is void', SCHEDULE_CONS[1]],
  },
  long_step_gd: {
    rule: `${GD_L},\\quad h = \\text{Grimmer's pattern of length } t,\\ \\text{cycled}`,
    order: '$O(1/k)$',
    intuition:
      'A short pattern of steps, most of them below 2/L and one far above it (12/L in the ' +
      'pattern of length 7), repeated. On average the pattern moves further than any safe ' +
      'constant step, and the proved O(1/k) constant grows with that average.',
    pros: ['A larger proved constant than constant-step descent, from steps fixed in advance'],
    cons: [
      ...SCHEDULE_CONS,
      'Tuned for the convex worst case: along a direction of curvature L one period barely contracts',
    ],
  },
  ogm: {
    rule: `\\mathbf{y}_k = ${X}_{k-1} - \\tfrac1L\\,${G('k-1')},\\quad ${X}_k = \\mathbf{y}_k + \\tfrac{\\theta_{k-1} - 1}{\\theta_k}(\\mathbf{y}_k - \\mathbf{y}_{k-1}) + \\tfrac{\\theta_{k-1}}{\\theta_k}(\\mathbf{y}_k - ${X}_{k-1})`,
    order: '$O(1/k^2)$',
    intuition:
      'A gradient step of length 1/L to 𝐲ₖ, then two momentum terms: Nesterov’s, along ' +
      '𝐲ₖ − 𝐲ₖ₋₁, and a second one along the gradient step itself. Its worst-case bound is half ' +
      'of Nesterov’s, the best any method of this kind can achieve.',
    pros: [
      'Optimal worst case: $f(\\mathbf{x}_N) - f^\\star \\le L\\|\\mathbf{x}_0 - \\mathbf{x}^\\star\\|^2/(N+1)^2$, and some $f$ attains it',
    ],
    cons: [
      'Needs L, and the horizon N is fixed in advance (its last step differs)',
      'No linear rate on strongly convex f: the momentum keeps oscillating',
    ],
  },
  fista: {
    rule: `${X}_k = \\mathbf{y}_k - \\tfrac{1}{L_k}\\,\\nabla f(\\mathbf{y}_k),\\quad \\mathbf{y}_{k+1} = ${X}_k + \\tfrac{t_k - 1}{t_{k+1}}(${X}_k - ${X}_{k-1})`,
    order: '$O(1/k^2)$',
    intuition:
      'A gradient step from the extrapolated point 𝐲ₖ, with a momentum coefficient that grows ' +
      'toward 1. When the momentum starts to push uphill, it is reset to 0 (adaptive restart), ' +
      'which removes the oscillation of plain FISTA.',
    pros: [
      'No L needed with backtracking: Lₖ grows until the step decreases f enough',
      'With restart, linear convergence on strongly convex quadratics without knowing μ',
    ],
    cons: [
      'Not monotone without restart: f oscillates while the momentum is large',
      'Lₖ never decreases: an early large Lₖ keeps every later step short',
    ],
  },
  anderson_gd: {
    rule: `\\mathbf{c}^{\\star} = \\arg\\min_{\\mathbf{1}^{\\top}\\mathbf{c} = 1} \\big\\|\\textstyle\\sum_i c_i \\nabla f(${X}_i)\\big\\|,\\quad ${X}_k = \\textstyle\\sum_i c_i^{\\star}\\big(${X}_i - \\beta\\alpha\\nabla f(${X}_i)\\big)`,
    order: 'linear (local)',
    intuition:
      'Keep the last m + 1 iterates and mix them with the weights that make the mixed gradient ' +
      'smallest, then take one gradient step from the mix. On a quadratic this is GMRES: it ' +
      'ends in n + 1 steps.',
    pros: [
      'Often far faster than gradient descent, with one gradient per step and no line search',
      'Exact in n + 1 steps on a quadratic (GMRES)',
    ],
    cons: [
      'Solves ∇f = 0, so it can stop at a saddle point or a maximizer',
      'f need not decrease; the least-squares weights become ill-conditioned near the end',
    ],
  },
  arc: {
    rule: `\\mathbf{s}_k = \\arg\\min_{\\mathbf{s}}\\ \\nabla f_k^{\\top}\\mathbf{s} + \\tfrac12\\,\\mathbf{s}^{\\top}\\nabla^2 f_k\\,\\mathbf{s} + \\tfrac{\\sigma_k}{3}\\|\\mathbf{s}\\|^3,\\quad ${X}_k = ${X}_{k-1} + \\mathbf{s}_k \\text{ if } \\rho_k \\ge \\eta_1`,
    order: 'superlinear',
    intuition:
      'Newton’s quadratic model plus a cubic penalty on long steps. Its global minimizer solves ' +
      '(∇²f + λI)𝐬 = −∇f with λ = σ‖𝐬‖: an implicit trust region of radius λ/σ. σ grows when ' +
      'the model mispredicts and shrinks when it predicts well.',
    pros: [
      'Escapes saddle points: negative curvature makes the cubic model’s minimizer move away',
      'Best worst-case complexity of second-order methods: $O(\\varepsilon^{-3/2})$ evaluations',
    ],
    cons: ['An eigenvalue problem per step for the global model minimizer'],
  },
  reg_newton: {
    rule: `${X}_k = ${X}_{k-1} - \\big(\\nabla^2 f(${X}_{k-1}) + \\lambda_k I\\big)^{-1}${G('k-1')},\\quad \\lambda_k = \\sqrt{H_k\\,\\|${G('k-1')}\\|}`,
    order: 'superlinear',
    intuition:
      'Newton’s step with the Hessian shifted by λ, which is large where ∇f is large (a short, ' +
      'gradient-like step) and vanishes as ∇f → 0 (Newton’s step). AdaN doubles H until the ' +
      'step decreases f enough.',
    pros: [
      'Global O(1/k²) on convex f with one linear solve per step and no line search',
      'Superlinear near a minimizer of a strongly convex f',
    ],
    cons: [
      'The theory is for convex f: on a nonconvex f it can converge to a saddle point or a maximizer',
    ],
  },
  nelder_mead: {
    rule: `${X}_r = \\bar{${X}} + \\rho\\,(\\bar{${X}} - ${X}_{n+1}),\\quad \\bar{${X}} = \\tfrac1n \\textstyle\\sum_{i \\le n} ${X}_i`,
    order: 'no general rate',
  },
  powell: {
    rule: `${X}_i = ${X}_{i-1} + \\alpha_i \\mathbf{u}_i,\\quad \\alpha_i = \\arg\\min_{\\alpha} f(${X}_{i-1} + \\alpha\\,\\mathbf{u}_i),\\quad \\mathbf{u}_{\\text{new}} = ${X}_n - ${X}_0`,
    order: 'no rate proven',
  },
  hooke_jeeves: {
    rule: `\\mathbf{p} = \\mathbf{b}_k + (\\mathbf{b}_k - \\mathbf{b}_{k-1}),\\quad \\text{explore } \\mathbf{p} \\pm h\\,\\mathbf{e}_i`,
    order: 'linear at best',
  },
  compass_search: {
    rule: `${X}_k = ${X}_{k-1} + \\Delta_k \\mathbf{d},\\ \\mathbf{d} \\in \\{\\pm\\mathbf{e}_i\\},\\quad \\Delta_{k+1} = \\theta\\,\\Delta_k \\text{ if no } \\mathbf{d} \\text{ decreases } f`,
    order: 'linear at best',
  },
};

/**
 * A long rule set on several centered lines: its top-level parts (separated by `,\quad`) go
 * one per line, so the card's display formula fits the 380 px column without scrolling.
 */
export function stackRule(rule: string, max = 64): string {
  const parts = rule.split(',\\quad ');
  if (rule.length <= max || parts.length < 2) return rule;
  return `\\begin{gathered} ${parts.join(' \\\\ ')} \\end{gathered}`;
}

/** The registered method with the lab's notation for its card (the port's doc otherwise). */
export function withLabDoc(m: RegisteredMethod): RegisteredMethod {
  const local = LAB_DOCS[m.spec.id];
  if (!local) return m;
  const base = m.doc ?? { rule: '', intuition: m.spec.summary };
  return {
    ...m,
    doc: {
      ...base,
      ...local,
      rule: stackRule(local.rule),
      intuition: local.intuition ?? base.intuition,
      pros: local.pros ?? base.pros,
      cons: local.cons ?? base.cons,
      // The "This step" panel shows the live quantities; the card keeps the textbook page.
      quantities: [],
    },
  };
}
