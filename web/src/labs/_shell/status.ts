import type { Params, Result } from '../../core/types';
import { int, sci, sig } from '../../core/format';

export interface RunStatus {
  tone: 'good' | 'warn' | 'bad';
  /**
   * Status word first, then the count with its unit: "converged · 596 iterations". This is the
   * text form for a place that shows the status as words (a pill, a table cell); the unit is never
   * abbreviated.
   */
  short: string;
  /** A plain sentence ("Converged in 204 iterations", "Converged in 47 moves — 2-optimal"). */
  long: string;
  /**
   * The count alone ("204"), for compact badges: the icon plus the count ("✓ 204"), with `long`
   * as the badge's accessible text and tooltip. One form for every lab's run chips.
   */
  count?: string;
  /**
   * A word the badge sets before the count of a converged run ("2-opt" → "✓ 2-opt 47"), from
   * `StatusWording.converged`. Absent for other runs and for labs without a qualifier.
   */
  qualifier?: string;
  /** Shape that carries the status without color: ✓ converged, ◷ budget, △ diverged/stopped. */
  icon: '✓' | '◷' | '△' | '×';
}

/** What one step of a family is called (`['trial', 'trials']` for line searches). */
export type IterationNoun = readonly [singular: string, plural: string];

/**
 * How a lab words its runs' status: the unit of one step and, optionally, what convergence means
 * there. Combinatorial 2-opt: `{ noun: ['move', 'moves'], converged: { badge: '2-opt', long:
 * '2-optimal' } }` → badge "✓ 2-opt 47", sentence "Converged in 47 moves — 2-optimal".
 */
export interface StatusWording {
  noun?: IterationNoun;
  converged?: { badge: string; long: string };
}

/** An `IterationNoun` (the older third argument) or a full `StatusWording`. */
export type StatusWordingArg = IterationNoun | StatusWording | undefined;

function wordingOf(arg: StatusWordingArg): Required<Pick<StatusWording, 'noun'>> & StatusWording {
  if (!arg) return { noun: ['iteration', 'iterations'] };
  if (Array.isArray(arg)) return { noun: arg as unknown as IterationNoun };
  const w = arg as StatusWording;
  return { ...w, noun: w.noun ?? ['iteration', 'iterations'] };
}

/**
 * A method name without its parenthetical variant ("Brent (zeroin)" → "Brent"), for chips and
 * lanes that have no room for it. The full name stays in the tooltip and accessible name.
 */
export function shortMethodName(name: string): string {
  return name.replace(/\s*\(.*\)\s*$/, '') || name;
}

/**
 * The first word of the short name ("Newton–Raphson" → "Newton", "Conjugate gradient" →
 * "Conjugate", "L-BFGS" stays whole): the next step down when the short names do not fit.
 */
export function methodWord(name: string): string {
  return shortMethodName(name).split(/[\s–]+/)[0] || name;
}

/** Plain-language status of a run (Python messages are terse). */
export function describeResult(
  result: Result,
  error?: string,
  wording?: StatusWordingArg,
): RunStatus {
  const { noun, converged: conv } = wordingOf(wording);
  if (error)
    return {
      tone: 'bad',
      short: 'invalid input',
      long: `Could not run: ${error}`,
      icon: '×',
      count: 'error',
    };
  const n = int(result.nIter);
  const its = result.nIter === 1 ? noun[0] : noun[1];
  const counted = `${n} ${its}`;
  if (result.converged)
    return {
      tone: 'good',
      short: `converged · ${counted}`,
      long: `Converged in ${counted}${conv ? ` — ${conv.long}` : ''}`,
      icon: '✓',
      count: n,
      qualifier: conv?.badge,
    };
  const m = result.message.toLowerCase();
  // An epoch budget ("completed 20 epochs (800 updates); ‖∇f‖ > gtol") is a budget stop too.
  if (
    m.includes('max_iter') ||
    m.includes('budget') ||
    m.includes('maximum number') ||
    /^completed \d+ epochs?\b/.test(m)
  )
    return {
      tone: 'warn',
      short: `budget · ${counted}`,
      long: `Stopped at the ${n}-${noun[0]} budget without converging`,
      icon: '◷',
      count: n,
    };
  // Divergence has three different causes; say which one, in words.
  const why = divergenceReason(result.message);
  if (why)
    return {
      tone: 'bad',
      short: `diverged · ${counted}`,
      long: `Diverged after ${n} ${its} (${why})`,
      icon: '△',
      count: n,
    };
  if (/\bcycl/.test(m))
    return {
      tone: 'warn',
      short: `cycled · ${counted}`,
      long: `Stopped after ${n} ${its}: the iterates cycle (${evidence(result.message)})`,
      icon: '△',
      count: n,
    };
  if (/\bstall/.test(m))
    return {
      tone: 'warn',
      short: `stalled · ${counted}`,
      long: `Stalled after ${n} ${its}: ${evidence(result.message)}`,
      icon: '△',
      count: n,
    };
  return {
    tone: 'warn',
    short: `stopped · ${counted}`,
    long: `Stopped after ${n} ${its}: ${evidence(result.message)}`,
    icon: '△',
    count: n,
  };
}

/**
 * Why a run diverged, from the Python message, or null when it did not:
 * a bound test on the iterates (`diverged: |x| = 2.38e+13 > 1.5e+12`, every value finite), an
 * increase of f, or a non-finite value (NaN, ±∞, overflow).
 */
export function divergenceReason(message: string): string | null {
  const m = message.toLowerCase();
  if (/diverged:\s*(\|x\||‖x‖|the iterates diverge)|iterates diverge: ‖x/.test(m))
    return '|x| grew past the divergence bound';
  if (/diverged:\s*f\(x\)\s*[−-]\s*f\(x0\)/.test(m)) return 'f grew past the divergence bound';
  if (/non-finite|not finite|\bnan\b|overflow|[-+]?\binf\b/.test(m)) return 'non-finite value';
  if (/diverg/.test(m)) {
    const tail = /diverged?:\s*(.+)$/i.exec(message)?.[1];
    return tail ? evidence(tail) : 'the iterates left the region where the method is defined';
  }
  return null;
}

/**
 * Stopping thresholds a Python message may name by id: tolerances, and the floors and bounds that
 * end a run (simulated annealing's freezing temperature `T_min`, a line search's `t_min`, AAA's
 * `sigma_min`, a trust region's `max_radius`). Brand §2: a threshold is stated as a number.
 */
const TOL_IDS = [
  'gtol',
  'xtol',
  'ftol',
  'ptol',
  'rtol',
  'atol',
  'tol',
  'T_min',
  't_min',
  'sigma_min',
  'max_radius',
];

/** A number token of a Python message: `0.001`, `-3`, `1e-10`, `2.5E+03`. */
const NUM = String.raw`[-+]?\d+\.?\d*(?:e[-+]?\d+)?`;
const NUM_TOKEN = new RegExp(String.raw`(?<![\w.^])${NUM}(?![\w.])`, 'gi');
/** `a ≤ b` (also `<`, `>`, `≥`, `<=`, `>=`), each side a number token. */
const COMPARISON = new RegExp(
  String.raw`(?<![\w.^])(${NUM})(\s*(?:≤|≥|<=|>=|<|>)\s*)(${NUM})(?![\w.])`,
  'gi',
);

/** True when a token is set in scientific notation on its own (e, ≥ 10⁵, or below 10⁻³). */
function needsSci(tok: string): boolean {
  const v = Number(tok);
  return /e/i.test(tok) || Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-3);
}

/** Scientific notation with an exact power of ten written alone: 1e-8 → 10⁻⁸, 2e-8 → 2×10⁻⁸. */
function sciText(v: number, digits: number): string {
  return sci(v, digits).replace(/^(−?)1×(?=10)/, '$1');
}

/**
 * The Python message as typeset evidence: numbers as 1.3×10⁻¹¹ (U+2212 minus), threshold ids
 * replaced by their values when `params` has them ("≤ gtol" → "≤ 10⁻⁸"), `max_iter=1000` →
 * "the 1,000-iteration budget". Both sides of a comparison use one notation: when either side
 * needs ×10ⁿ and the two are within two decades, both get it ("T = 9.98×10⁻⁴ ≤ 10⁻³", never
 * "≤ 0.001"); far-apart values keep their own form ("‖∇f‖ = 0.136 > 10⁻⁶"). Used under the
 * status sentence in the MethodCard.
 */
export function evidence(message: string, params: Params = {}): string {
  let s = message.replace(
    /max_iter\s*=\s*(\d+)/g,
    (_, k: string) => `the ${int(Number(k))}-iteration budget`,
  );
  // "≤ tol = 1e-10" → "≤ 1e-10": the threshold is stated as a number, never by its id.
  s = s.replace(new RegExp(`\\b(?:${TOL_IDS.join('|')})\\s*=\\s*(?=[-+−]?\\d|\\.\\d)`, 'g'), '');
  // Numbers on either side of a comparison share a notation.
  const forced = new Set<number>();
  for (const m of s.matchAll(COMPARISON)) {
    const [, a, op, b] = m;
    const [va, vb] = [Math.abs(Number(a)), Math.abs(Number(b))];
    // Close values (within two decades) are the ones a reader compares digit by digit.
    const close =
      Number.isFinite(va) &&
      Number.isFinite(vb) &&
      va > 0 &&
      vb > 0 &&
      Math.abs(Math.log10(va / vb)) <= 2;
    if (close && (needsSci(a) || needsSci(b))) {
      forced.add(m.index);
      forced.add(m.index + a.length + op.length);
    }
  }
  s = s.replace(NUM_TOKEN, (tok, offset: number) => {
    const v = Number(tok);
    if (!Number.isFinite(v)) return tok;
    if (forced.has(offset) || needsSci(tok)) return sciText(v, 3);
    return tok.includes('.') ? sig(v, 4) : tok.replace(/^-/, '−');
  });
  for (const id of TOL_IDS) {
    const v = params[id];
    if (typeof v !== 'number') continue;
    s = s.replace(new RegExp(`\\b${id}\\b(?!\\s*=)`, 'g'), sciText(v, 2));
  }
  return s.replace(/ - /g, ' − ');
}
