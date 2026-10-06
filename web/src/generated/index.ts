/**
 * Guarded, lazy access to `numopt export` output (registry + problem metadata). Every loader resolves to empty data when the
 * file is missing, so the build never depends on the Python side having been run.
 */
import { methodSpecFromJson, problemMetaFromJson } from '../core/json';
import type { MethodSpec, ProblemMeta } from '../core/types';

type Loader = () => Promise<unknown>;

const registryFiles = import.meta.glob('./registry.json', { import: 'default' }) as Record<
  string,
  Loader
>;
const problemFiles = import.meta.glob('./problems.json', { import: 'default' }) as Record<
  string,
  Loader
>;

let registryCache: Promise<MethodSpec[]> | null = null;
let problemCache: Promise<ProblemMeta[]> | null = null;

/** Python method specs (empty when not exported yet). */
export function loadGeneratedRegistry(): Promise<MethodSpec[]> {
  registryCache ??= (async () => {
    const load = registryFiles['./registry.json'];
    if (!load) return [];
    const raw = (await load()) as Record<string, unknown>[];
    return raw.map(methodSpecFromJson);
  })();
  return registryCache;
}

/** Python problem metadata (empty when not exported yet). */
export function loadGeneratedProblems(): Promise<ProblemMeta[]> {
  problemCache ??= (async () => {
    const load = problemFiles['./problems.json'];
    if (!load) return [];
    const raw = (await load()) as Record<string, unknown>[];
    return raw.map(problemMetaFromJson);
  })();
  return problemCache;
}

// Fixtures are test data: tests/parity.test.ts reads src/generated/fixtures/*.json from disk.
// They are deliberately NOT globbed here, so they never end up in the app bundle.
