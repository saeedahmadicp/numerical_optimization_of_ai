/** Which lab a problem kind opens in (pure; shared by src/labs/index.ts and the search index). */

/**
 * The open lab named after the kind (or showing a family of that name), else the first open lab
 * that uses the kind, else the first such lab.
 */
export function pickLabForKind<
  L extends {
    id: string;
    families: readonly string[];
    problemKinds: readonly string[];
    status: string;
  },
>(labs: readonly L[], kind: string): L | undefined {
  const all = labs.filter((l) => l.problemKinds.includes(kind));
  const open = all.filter((l) => l.status === 'open');
  const named = (l: L) => l.id === kind.replace(/_/g, '-') || l.families.includes(kind);
  return open.find(named) ?? open[0] ?? all.find(named) ?? all[0];
}
