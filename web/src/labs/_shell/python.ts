/**
 * The Python call that reproduces a run in the reference package, e.g.
 *   numopt.run("bfgs", problems.get("rosenbrock"), x0=[-1.2, 1.0], gtol=1e-10)
 * Only parameters that differ from the registered defaults are written.
 */
import type { ParamSpec, ParamValue, Params } from '../../core/types';

export interface PythonCallOptions {
  /** Problem id in `numopt.problems` (omit for a bare callable placeholder `f`). */
  problem?: string;
  x0?: number | readonly number[];
  bracket?: readonly number[];
  seed?: number;
  /** Parameter values of the run (defaults are dropped when `specs` is given). */
  params?: Params;
  specs?: readonly ParamSpec[];
}

/** Python's repr of a float: `1.0`, `0.002`, `1e-08`, `-1.2`. */
export function pyFloat(v: number): string {
  if (Number.isNaN(v)) return 'float("nan")';
  if (!Number.isFinite(v)) return v > 0 ? 'float("inf")' : '-float("inf")';
  if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`;
  const s = String(v);
  const m = /^(-?[\d.]+)e([+-])(\d+)$/.exec(s);
  if (m) return `${m[1]}e${m[2]}${m[3].padStart(2, '0')}`;
  // Python switches to exponent form below 1e-4.
  if (Math.abs(v) < 1e-4) {
    const [mant, exp] = v.toExponential().split('e');
    const e = Number(exp);
    return `${mant.replace(/\.?0+$/, '')}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  }
  return s;
}

export function pyValue(v: ParamValue, kind?: ParamSpec['kind']): string {
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map((x) => pyFloat(x)).join(', ')}]`;
  if (kind === 'int' && Number.isInteger(v)) return String(v);
  return pyFloat(v);
}

const same = (a: ParamValue, b: ParamValue) => JSON.stringify(a) === JSON.stringify(b);

export function pythonCall(methodId: string, o: PythonCallOptions = {}): string {
  const args = [
    JSON.stringify(methodId),
    o.problem ? `problems.get(${JSON.stringify(o.problem)})` : 'f',
  ];
  if (o.x0 !== undefined)
    args.push(`x0=${typeof o.x0 === 'number' ? pyFloat(o.x0) : pyValue([...o.x0])}`);
  if (o.bracket) args.push(`bracket=(${o.bracket.map(pyFloat).join(', ')})`);
  if (o.seed !== undefined) args.push(`seed=${o.seed}`);
  const specs = new Map((o.specs ?? []).map((p) => [p.name, p]));
  for (const [k, v] of Object.entries(o.params ?? {})) {
    const spec = specs.get(k);
    if (spec && same(spec.default, v)) continue;
    args.push(`${k}=${pyValue(v, spec?.kind)}`);
  }
  return `numopt.run(${args.join(', ')})`;
}

/** A runnable snippet (imports + the call + the result line). */
export function pythonSnippet(methodId: string, o: PythonCallOptions = {}): string {
  const lines = ['import numopt'];
  if (o.problem) lines.push('from numopt import problems');
  lines.push(`res = ${pythonCall(methodId, o)}`, 'print(res.converged, res.n_iter, res.x)');
  return lines.join('\n');
}
