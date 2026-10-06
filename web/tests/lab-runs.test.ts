// @vitest-environment jsdom
import { createElement, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import '../src/methods';
import { listProblems } from '../src/problems';
import type { LinalgProblem } from '../src/problems/linalg';
import { useLabRunsState } from '../src/labs/_shell/useLabRuns';
import type { MethodSelection } from '../src/labs/_shell/slots';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SELECTION: MethodSelection[] = [{ id: 'jacobi', slot: 0, params: {} }];
const OPTIONS = {};

describe('useLabRunsState', () => {
  let root: ReturnType<typeof createRoot> | null = null;
  afterEach(() => {
    act(() => root?.unmount());
    root = null;
  });

  it('keeps the runs stable when `defer` switches on after mount (no render loop)', () => {
    const problems = listProblems<LinalgProblem>('linalg');
    const small = problems.find((p) => p.n <= 4)!;
    const large = problems.find((p) => p.n > 4)!;
    let renders = 0;
    let seen: unknown[] = [];
    let pick: (p: LinalgProblem) => void = () => {};

    function Probe() {
      const [problem, setProblem] = useState(small);
      pick = setProblem;
      const { runs } = useLabRunsState(problem, SELECTION, OPTIONS, { defer: problem.n > 4 });
      // The same contract the labs rely on: `traces` memoized on `runs`.
      const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
      renders++;
      seen = traces;
      return null;
    }

    const el = document.createElement('div');
    root = createRoot(el);
    act(() => root!.render(createElement(Probe)));
    expect(seen.length).toBe(1);
    renders = 0;
    act(() => pick(large));
    // A render loop would throw "Too many re-renders" inside act.
    expect(renders).toBeLessThan(10);
    expect(seen.length).toBe(1);
    // Back to a small problem and to the large one again: no stale or empty runs.
    act(() => pick(small));
    act(() => pick(large));
    expect(seen.length).toBe(1);
  });
});
