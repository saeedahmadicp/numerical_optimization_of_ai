/* eslint-disable @typescript-eslint/consistent-type-imports -- ambient modules cannot import relative types */
// Virtual modules produced by web/vite/catalog.ts at build time.
declare module 'virtual:numopt/catalog' {
  const counts: import('./catalogTypes').CatalogCounts;
  export default counts;
}
declare module 'virtual:numopt/index' {
  export const methods: import('./catalogTypes').CatalogMethod[];
  export const problems: import('./catalogTypes').CatalogProblem[];
  export const research: import('./catalogTypes').CatalogResearch[];
}
declare module 'virtual:numopt/parity/*' {
  const cases: import('./catalogTypes').ParityCase[];
  export default cases;
}
declare module 'virtual:numopt/research' {
  export const studies: import('./catalogTypes').StudySummary[];
  export const about: { title: string; lede: string };
  export const loaders: Record<
    string,
    () => Promise<{ default: import('./catalogTypes').StudyDoc | null }>
  >;
}
declare module 'virtual:numopt/parity' {
  export const loaders: Record<
    string,
    () => Promise<{ default: import('./catalogTypes').ParityCase[] }>
  >;
}
