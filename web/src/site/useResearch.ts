/**
 * The research index module and the study documents, as module-cached promises read with React
 * 19 `use()`: the Research pages suspend under their route's PageLoading fallback and render once,
 * complete (no empty frame, no footer jump). Loading the index also starts the KaTeX stylesheet
 * the study HTML needs.
 */
import { use } from 'react';
import { preloadKatex } from '../ui/katex';
import type { StudyDoc } from '../app/catalogTypes';

import type * as Research from 'virtual:numopt/research';

export type ResearchModule = typeof Research;

let research: Promise<ResearchModule> | null = null;
const studies = new Map<string, Promise<StudyDoc | null>>();

export function loadResearch(): Promise<ResearchModule> {
  if (!research) {
    void preloadKatex();
    research = import('virtual:numopt/research');
  }
  return research;
}

/** One study's document (null for an unknown id). */
export function loadStudy(id: string): Promise<StudyDoc | null> {
  let p = studies.get(id);
  if (!p) {
    p = loadResearch().then((mod) => {
      const load = mod.loaders[id];
      return load ? load().then((m) => m.default) : null;
    });
    studies.set(id, p);
  }
  return p;
}

/** The research index (suspends until it has loaded). */
export function useResearch(): ResearchModule {
  return use(loadResearch());
}

/** One study (suspends until it has loaded); null when there is no study with this id. */
export function useStudyDoc(id: string): StudyDoc | null {
  return use(loadStudy(id));
}
