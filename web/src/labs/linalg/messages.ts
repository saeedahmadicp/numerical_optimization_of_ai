/**
 * Python's messages in the lab's notation. The Python reference indexes from 0 and writes
 * subscripts with underscores ("zero pivot |a_00| = 0 ≤ τ", "a_22 = 0: …"); every caption and
 * formula of the lab is 1-based, so the symbols are rewritten to 1-based Unicode subscripts
 * ("|a₁₁|") before the shared `evidence` typesets the numbers. Copy actions keep the raw text.
 */
import type { Params, Result } from '../../core/types';
import { describeResult, evidence, type RunStatus } from '../_shell';

const SUB_DIGIT = '₀₁₂₃₄₅₆₇₈₉';
const SUB_LETTER: Record<string, string> = { i: 'ᵢ', j: 'ⱼ', k: 'ₖ' };
const sub = (n: number) =>
  String(n)
    .split('')
    .map((d) => SUB_DIGIT[Number(d)])
    .join('');

/**
 * `a_00` → `a₁₁`, `|u_22|` → `|u₃₃|`, `w_4` → `w₅`, `l_3j²` → `l₄ⱼ²`, `a_ij − a_ji` → `aᵢⱼ − aⱼᵢ`.
 * For a, u, r and l a digit run of even length with equal halves is a diagonal pair (both
 * shifted); w and d (and a digit run followed by a letter index) carry a single index.
 */
export function oneBased(message: string): string {
  let s = message.replace(
    /(?<![\p{L}\p{N}_])([a-zA-Z])_(\d+)([ijk]?)/gu,
    (_, letter: string, digits: string, tail: string) => {
      const h = digits.length / 2;
      // Thomas's w_i and Cholesky's d_c carry one index; a, u, r and l carry a diagonal pair.
      const pair =
        !'wd'.includes(letter) &&
        !tail &&
        digits.length % 2 === 0 &&
        digits.slice(0, h) === digits.slice(h);
      const idx = pair
        ? `${sub(Number(digits.slice(0, h)) + 1)}${sub(Number(digits.slice(h)) + 1)}`
        : sub(Number(digits) + 1);
      return `${letter}${idx}${tail ? SUB_LETTER[tail] : ''}`;
    },
  );
  s = s.replace(
    /(?<![\p{L}\p{N}_])([a-zA-Z])_([ijk]{1,2})(?![\p{L}\p{N}])/gu,
    (_, letter: string, idx: string) =>
      `${letter}${idx
        .split('')
        .map((c) => SUB_LETTER[c])
        .join('')}`,
  );
  return s;
}

/** A Python message as typeset, 1-based evidence. */
export const msgText = (message: string, params: Params = {}) =>
  evidence(oneBased(message), params);

/** `describeResult` on the 1-based message. */
export const describe = (result: Result, error?: string): RunStatus =>
  describeResult({ ...result, message: oneBased(result.message) }, error && oneBased(error));
