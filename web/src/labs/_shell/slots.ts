import type { ParamSpec, ParamValue, Params } from '../../core/types';
import { MAX_SERIES } from '../../ui/colors';

export interface MethodSelection {
  id: string;
  /** Color slot (stable for the life of the selection: color follows the method, not its rank). */
  slot: number;
  params: Params;
}

/** Lowest color slot not used by the selection, or −1 when all four are taken. */
export function firstFreeSlot(sel: readonly MethodSelection[]): number {
  for (let s = 0; s < MAX_SERIES; s++) if (!sel.some((m) => m.slot === s)) return s;
  return -1;
}

/** Coerce one URL/user value to a ParamSpec (clamped to its range); `undefined` = drop it. */
export function coerceParam(spec: ParamSpec, v: ParamValue): ParamValue | undefined {
  switch (spec.kind) {
    case 'float':
    case 'int': {
      const n = typeof v === 'number' ? v : Number(v);
      if (!Number.isFinite(n)) return undefined;
      let x = spec.kind === 'int' ? Math.round(n) : n;
      if (spec.min !== null) x = Math.max(spec.min, x);
      if (spec.max !== null) x = Math.min(spec.max, x);
      return x;
    }
    case 'bool':
      return typeof v === 'boolean' ? v : v === 'true' ? true : v === 'false' ? false : undefined;
    case 'choice':
      return spec.choices.includes(String(v)) ? String(v) : undefined;
    default:
      return v;
  }
}

/** The subset of a registered method that sanitizing needs. */
export interface SelectableMethod {
  spec: { id: string; params: ParamSpec[] };
}

/**
 * Make a selection safe to run and to draw (URLs are user input):
 * - drop ids that are not in `available` (reported in `dropped`) and duplicate ids (with
 *   `allowDuplicates`, a method may appear twice — each entry is then identified by its slot);
 * - keep at most four methods;
 * - give every method its own color slot: a slot that is out of range or already taken is
 *   replaced by the first free one (so two paths never share a color);
 * - keep only declared params, coerced to their kind and clamped to their range.
 */
export function sanitizeSelection(
  sel: readonly MethodSelection[],
  available: readonly SelectableMethod[],
  { allowDuplicates = false }: { allowDuplicates?: boolean } = {},
): { value: MethodSelection[]; dropped: string[] } {
  const byId = new Map(available.map((m) => [m.spec.id, m]));
  const value: MethodSelection[] = [];
  const dropped: string[] = [];
  const pending: MethodSelection[] = [];
  const pos = new Map<MethodSelection, number>();
  for (const [i, s] of sel.entries()) {
    const m = byId.get(s.id);
    if (!m) {
      if (!dropped.includes(s.id)) dropped.push(s.id);
      continue;
    }
    if (
      !allowDuplicates &&
      (value.some((v) => v.id === s.id) || pending.some((v) => v.id === s.id))
    )
      continue;
    if (value.length + pending.length >= MAX_SERIES) break;
    const params: Params = {};
    for (const p of m.spec.params) {
      if (!(p.name in s.params)) continue;
      const v = coerceParam(p, s.params[p.name]);
      if (v !== undefined) params[p.name] = v;
    }
    const slotOk =
      Number.isInteger(s.slot) &&
      s.slot >= 0 &&
      s.slot < MAX_SERIES &&
      !value.some((v) => v.slot === s.slot);
    const entry = { id: s.id, slot: slotOk ? s.slot : -1, params };
    pos.set(entry, i);
    (slotOk ? value : pending).push(entry);
  }
  // Second pass, after every valid slot is claimed, so a collision never steals a later slot.
  for (const p of pending) {
    const entry = { ...p, slot: firstFreeSlot(value) };
    pos.set(entry, pos.get(p) ?? 0);
    value.push(entry);
  }
  // Keep the original order of the selection.
  value.sort((a, b) => (pos.get(a) ?? 0) - (pos.get(b) ?? 0));
  return { value, dropped };
}
