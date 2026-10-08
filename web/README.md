# numopt web — the visualizer

A static, server-less React app (Vite + React 19 + TypeScript strict) with one **Lab** per topic.
Every lab is built on a shared platform: a design system (`src/ui`), rendering primitives
(`src/viz`), a playback engine (`src/play`), a lab layout (`src/labs/_shell`) and TS ports of the
Python methods (`src/methods`) that are parity-tested against the Python fixtures.

```bash
npm install
npm run dev          # http://localhost:5173  (#/ home, #/labs, #/lab/<id>, #/methods, #/method/<id>, #/research, #/research/<study>, #/python, #/dev)
npm run build        # tsc -b && vite build  → dist/ (GitHub Pages ready: base './', hash router)
npm run lint         # eslint (flat config, react-hooks v7 / React Compiler rules), 0 warnings allowed
npm test             # vitest: unit tests + parity harness
npm run e2e          # playwright smoke tests (builds + previews on :4173; E2E_PORT=… for a private port)
npm run gen          # ../.venv/bin/numopt export src/generated  (registry, problems, fixtures)
npm run format       # prettier
```

## File layout

```
src/
  core/        types.ts (Step, Result, Problem, LinearProgram, LinearSystem, Dataset, MethodSpec, ParamSpec)
               rng.ts (Mulberry32, bit-identical to numopt.core.rng)   json.ts (export loader: "inf"/"-inf"/null)
               registry.ts (registerMethod / getMethod / listMethods / runMethod / param.*)
               linalg.ts (dot, axpy, norm, matvec, solve [LU, partial pivot], cholesky, eigSym2, …)
               format.ts (sig, sci with unicode superscripts, sigFixed, vec, tick, powTen)
  generated/   output of `numopt export` — never hand-edit (see its README)
  methods/     <pkg>/<module>.ts — TS ports, mirroring src/numopt/<pkg>/<module>.py   (index.ts globs them)
  problems/    <kind>.ts — TS ports of the problem library (same ids)              (index.ts globs them)
  ui/          tokens.css, global.css, colors.ts, theme.ts, components/*
  viz/         useCanvas, scales, axes, Plot1D, Contour2D (+ worker), PathLayer, ConvergenceChart,
               IterationTable (+ columns), MatrixView, Surface3D (lazy three.js)
  play/        useTracePlayer, usePlayerKeyboard, timeline (pure math), reducedMotion
  labs/        index.ts (discovers the labs) · types.ts (LabMeta) · _shell/ (LabShell, blocks, presets,
               labState, python, useLabRuns, slots, status)
               <lab-id>/ meta.ts (always) + index.tsx (default export: the lab, once it exists) + anything else
  app/         App, router, useUrlState, catalog (build-time counts + lazy index), Home + home/ (HeroShowcase,
               LabCard, previews/), LabsIndex, AppHeader + Search (lazy palette), Footer, Pages,
               DevPlayground + DevPrimitives (lazy)
  site/        the reference pages (each a lazy chunk): MethodsPage (#/methods), MethodPage (#/method/<id>:
               LiveRun, parity replay), ResearchPage + StudyPage (#/research, #/research/<id>), PythonPage,
               CodeBlock; pure helpers methodMeta.ts, parity.ts, liveMeasure.ts, runtime.ts (loads one
               family's ports on demand)
vite/          catalog.ts: the Vite plugin behind the `virtual:numopt/*` modules (see "App");
               research.ts: research READMEs → HTML at build time (marked + KaTeX in Node)
tests/         vitest: rng.test.ts (vs Python stream), core.test.ts, viz.test.ts, shell.test.ts, parity.test.ts, fixtures/
e2e/           playwright smoke tests
```

## How to build a lab

Every lab lives in its own folder, `src/labs/<id>/`, and no shared file has to change to add one:

| File          | Content                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `meta.ts`     | `export default meta` — a `LabMeta` (`src/labs/types.ts`): `id` (= folder), `title`, `group` (`Equations` · `Optimization` · `Constrained & discrete` · `Numerical analysis` · `Data`), `problem` (LaTeX, shown in the syllabus and search), `pitch` (≤ 90 chars, name only methods that ship), `families`, `problemKinds`, optional `status: 'preview'`, `order`, `icon`. **Pre-created for all 16 labs.** |
| `index.tsx`   | `export default function MyLab()` — the lab component. Its presence turns the lab from _planned_ into _open_ (green pill on the home page, linked from search). Use `status: 'preview'` in `meta.ts` to keep a built lab unannounced.                                                                                                                                                                       |
| anything else | `setup.ts`, local components, CSS modules, tests — your folder, your files.                                                                                                                                                                                                                                                                                                                                 |

`src/labs/index.ts` discovers them with `import.meta.glob` (metas eager — they are tiny — components
lazy, each in its own chunk). Method counts come from `src/generated/registry.json` at build time
(`CATALOG.byFamily`), so the home page never loads a lab to count it. A planned lab's route shows
its problem, pitch, and the methods and test problems of its families from the registry.

1. **Port the methods** (`src/methods/<pkg>/<module>.ts`, same ids/params/trace semantics as Python — see
   "Porting a method" below) and **the problems** (`src/problems/<kind>.ts`). They register themselves at
   import time. A lab registers only what it needs, e.g. in `setup.ts`:
   `import.meta.glob('../../methods/roots/*.ts', { eager: true })` (or `import '../../methods'` for all).
2. **Write `index.tsx`** on the shell. `src/labs/unconstrained/UnconstrainedLab.tsx` is the reference; the
   skeleton is:

```tsx
import './setup';                                                       // registers methods + problems
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import { PlaybackBar } from '../../ui/components';
import { preloadKatex } from '../../ui/katex';
import { ConvergenceChart, Plot1D, type Overlay1D } from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { getLab } from '../index';
import { LabShell, RailSection, ProblemPicker, MethodSlots, MethodCard, RunSummary, useLabRunsState,
         useMethodSelection, useProblemState, type LabPreset, type MethodSelection } from '../_shell';

void preloadKatex();
const DEFAULT: MethodSelection[] = [{ id: 'bisection', slot: 0, params: {} }];   // module constants
const PRESETS: LabPreset[] = [
  { id: 'slow', title: 'Bisection halves, Newton squares', note: 'Same bracket, two rates.',
    problem: 'cubic', methods: [{ id: 'bisection', slot: 0, params: {} }, { id: 'newton', slot: 1, params: {} }] },
];

export default function RootsLab() {
  const problems = listProblems<MyProblem>('roots');
  const methods = listMethods('roots');
  const [problem, setProblem] = useProblemState(problems, 'cubic');      // ?p=, drops ?x0= on change
  const [sel, setSel] = useMethodSelection(methods, DEFAULT);            // ?m=, validated
  const { runs, pending } = useLabRunsState(problem, sel, { bracket: problem.bracket! }, { defer: false });
  const player = useTracePlayer(useMemo(() => runs.map((r) => r.result.trace), [runs]));  // stable array!
  usePlayerKeyboard(player);
  return (
    <LabShell
      lab={getLab('roots')!}
      presets={PRESETS}                                                  // "Try this" rail section
      pending={pending}                                                  // dims the stage, "Recomputing…"
      stageNotice={runs.length === 0 ? 'Add a method to compare — up to four run on one clock.' : undefined}
      controls={<>
        <RailSection title="Problem"><ProblemPicker problems={problems} value={problem.id} onChange={setProblem} /></RailSection>
        <RailSection title="Methods" actions={`${runs.length} / 4`}><MethodSlots available={methods} value={sel} onChange={setSel} /></RailSection>
      </>}
      stageTitle={<RunSummary runs={runs.map((r) => ({ name: r.method.spec.name, slot: r.sel.slot, result: r.result, error: r.error }))} />}
      stage={<Plot1D f={problem.f} domain={problem.domain} zeroLine overlays={overlaysAt(runs, player)} ariaLabel="…" />}
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        { id: 'conv', title: 'Convergence', height: 264, content: <ConvergenceChart series={…} t={player.t} yLabel="|f(xₖ)|" /> },
        { id: 'card', title: 'Method', grow: true, content:
            <MethodCard method={runs[0].method} slot={runs[0].sel.slot} step={…} result={runs[0].result}
                        error={runs[0].error} call={{ problem: problem.id, options: { bracket: … }, params: runs[0].sel.params }} /> },
      ]}
    />
  );
}
```

`useTracePlayer` restarts when its `traces` array changes identity — pass a memoized array (or a
module constant), never an inline literal, or it re-renders forever.

`LabShell` gives the header (breadcrumb, theme toggle, share button), a "Skip to visualization" link, the
control rail (with the lab's `<h1>`), the stage with the docked playback bar (marked
`data-player-scope`, so Home/End drive the player only there), the insights column, and the responsive
behavior: ≥1280px three columns; 900–1279px rail + stage on the first screen and the insights below
across the full width; <900px stage first, controls in a modal bottom sheet (focus moves to its close
button, the page behind is `inert`, Tab stays inside, Esc/close returns focus to the Controls button;
half height so the plot stays visible, drag up or press the handle for full height). On phones the
"Controls" button is docked into the playback bar (icon-only below 480 px), so it never covers the
chart. The lab title is set in Newsreader; the problem formula under the picker is the rail's anchor
(20 px KaTeX, shrunk to fit with `<Formula fit>`).

## Platform APIs

### Core (`src/core`)

| API                                                                                                                                                                 | Notes                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Step {k, x, fun, gradNorm, stepSize, info}`, `Result {method, x, fun, converged, message, nIter, nFev, nGev, nHev, trace, extra}`                                  | camelCase mirror of Python. `info` keys stay **snake_case exactly as documented in the Python module docstring** (`bracket`, `direction`, `simplex`, `trials`, `tableau`, …).                                                                                                                                                                |
| `registerMethod(spec, fn, doc?)`                                                                                                                                    | `spec`: `{id, family, name, params, needs, order, summary, references, deterministic, tags}` (omitted fields get Python defaults). `fn(problem, {x0?, bracket?, seed?, ...params}) → Result`. `doc` (TS-only, for the MethodCard): `{rule (LaTeX), intuition, order?, pros?, cons?, quantities?: [{tex, key: 'stepSize' \| 'info.<key>'}]}`. |
| `getMethod(id)`, `hasMethod(id)`, `listMethods(family?)`, `runMethod(id, problem, params)`, `defaults(spec)`                                                        | `runMethod` rejects unknown params like Python's `run()`.                                                                                                                                                                                                                                                                                    |
| `param.float/int/bool/choice(name, default, {min, max, log, help, label, tex})`                                                                                     | ParamSpec builders; read like the Python `@register`. `label` ("Step size") and `tex` (`\\alpha`) are TS-only display fields; the UI falls back to the humanized `name`.                                                                                                                                                                     |
| `new Rng(seed)` → `random, uniform, normal, integers, permutation, choice`                                                                                          | Bit-identical to `numopt.core.rng.Rng` (tested against 1,400+ Python values). Draw in the same order as Python.                                                                                                                                                                                                                              |
| `parseJson`, `reviveNumbers`, `resultFromJson`, `methodSpecFromJson`, `problemMetaFromJson`, `fixtureCaseFromJson`, `linearProgramFromJson`, `linearSystemFromJson` | `"inf"`/`"-inf"` → ±Infinity; `null` (Python NaN/None) stays `null`.                                                                                                                                                                                                                                                                         |
| `dot axpy add sub scale norm normInf matvec matmul transpose identity outer luFactor luSolve solve cholesky choleskySolve eigSym2`                                  | Small dense helpers on plain `number[]`; inputs never mutated; `solve`/`cholesky` return `null` on breakdown.                                                                                                                                                                                                                                |
| `sig sci sigFixed vec vecFixed tick powTen superscript int`                                                                                                         | `sci(1.2e-8) = "1.2×10⁻⁸"`, U+2212 minus, `—` for null/NaN, `∞`. `sigFixed`/`vecFixed` keep trailing zeros (and a sign column) for aligned table columns.                                                                                                                                                                                    |
| `loadGeneratedRegistry()`, `loadGeneratedProblems()` (`src/generated`)                                                                                              | Lazy + guarded: resolve to `[]` when the export has not been run.                                                                                                                                                                                                                                                                            |
| `addProblem(kind, p)`, `getProblem(id)`, `listProblems(kind)` (`src/problems/registry`)                                                                             | Same ids and kinds as the Python problem registry.                                                                                                                                                                                                                                                                                           |

### Design system (`src/ui`)

Components (`src/ui/components`, all keyboard accessible with solid 2-px focus rings and ARIA):
`Button`, `IconButton` (tooltip + `aria-keyshortcuts`; `tooltipSide`; the tip is not linked as a
description because it repeats the name), `Select`/`Combobox` (searchable listbox, groups,
descriptions), `Slider` (linear/log, integer, arrows/PageUp/Home/End; log sliders expose a 0–1000
position with the real value in `aria-valuetext`), `NumberField` (accepts `1e-8`, ↑/↓ nudges, Esc
reverts), `Toggle` (switch), `SegmentedControl` (radiogroup with roving tabindex; per-option `tooltip`,
e.g. a swatch-only method picker), `Tabs` + `TabPanel` (`<TabPanel idBase id>` writes the role, ids and
labelling; `tabPanelProps` for custom panels), `Tooltip` (`side`, flips when it would leave the
viewport; `describe={false}` when it only repeats the trigger's name), `Kbd`, `Panel`/`Card`, `Badge`
(status tones always with text), `Formula` (KaTeX, memoized, `display`; KaTeX loads lazily in its own
chunk — `preloadKatex()` from `src/ui/katex` starts it early), `MethodChip`/`Swatch` (remove button
with a 32 × 32 hit area), `CopyButton` (inline "Copied" swap, no toast; `copyText`/`useCopied` in `src/ui/copy.ts`), `Menu` (overflow
menu button: action / copy / link items, arrow keys, Esc), `Num` (typeset number: tabular, U+2212, ×10ⁿ, `—`,
`∞`; `mode="int"` groups digits for counts only), `Formula` also takes `fallback` (plain-text stand-in while
KaTeX loads, no layout shift) and `fit` (shrink a display formula to its box, down to 70 %; below that it switches to a line-breaking
`\displaystyle` layout instead of clipping; measured again when the KaTeX fonts load and when the box
width changes), `ParamControls` (builds a control per `ParamSpec`: slider+field for float/int —
log when `spec.log`, e-notation in the field —, switch for bool, segments/select for choice, fields for
vector; shows `spec.tex` + `spec.label`), `PlaybackBar` (`displayK` names a step other than `player.k` in
the readout, e.g. switched halfway through a cross-fade), `Select` (listbox type-ahead when it has no
search box), `ToastProvider`/`useToast`, `Icon`. Import
components from their files (not the barrel) on pages that must stay small, like the home page.

Theme: `useTheme()` → `{preference: 'system'|'light'|'dark', mode, setPreference}`; `systemTheme()`;
the header toggle switches between the two resolved themes (picking the OS theme stores `'system'`
again). `useChartColors()` → the chart tokens (surface, grid, axis, tick, iso, halo, series[4], fonts)
for canvas drawing, recomputed when the theme changes. Open `#/dev` on the dev server to see every component live (the catalog is not in production builds).

### Visualization (`src/viz`)

| Component                                                                                                                                                           | Use                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `useCanvas(draw)`                                                                                                                                                   | HiDPI canvas: tracks CSS size (ResizeObserver) + devicePixelRatio, redraws once per frame after every render and when fonts load. `draw(ctx, {width, height, dpr})` gets a context scaled to CSS px. A static layer passes `{ deps: [...] }` to repaint only when those values (or the size) change, not on every player frame. `useElementSize(ref)` for the same measurement without a canvas.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `linearScale`, `logScale`, `linearTicks`, `logTicks`, `niceDomain`, `logExtent`, `crisp`                                                                            | `crisp(v, dpr)` centers a 1-device-pixel line (0.5 CSS px at 2×).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `drawAxes(ctx, {x, y, frame, colors, dpr, xLabel, yLabel, inset, halo, xInteger})`                                                                                  | Hairline grid, tabular (monospace) tick labels, powers of ten on log axes. `inset` draws labels inside the frame, 8 px clear of the edges, with a 3-px `--chart-halo` outline (`halo`) and single-letter axis names in italics.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `<Plot1D f domain overlays logY onPick hideCurve xName yName>`                                                                                                      | Adaptive sampling. Overlays: `interval` (shaded bracket + label), `area` (∫ shading of f or any g), `segment` (optional arrowhead), `arrow`, `tangent`, `parabola`, `curve` (another function: interpolant, model), `polyline` (`fill: 'under'` for trapezoids, `dots`), `vline`/`hline` (labelled markers), `point` (dot / ring / + cross, labelled), `text`. Labels are strings or math runs (`[mathVar('x'), mathSub('k')]`). `slot` picks a method color; text stays in text tokens.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `<Contour2D f domain cacheKey paths t minima minimaLabels start overlays offViewLabels onPick overlay overlayAfter axisLabels valueLabel initialView onFieldReady>` | Filled bands + iso-lines (marching squares in a Web Worker). Levels are fixed per problem (`levelSpecFor` over the full `domain`, log spacing chosen automatically for steep functions), so one f keeps one color in every view and pan/zoom only re-rasterize. Wheel/drag pan & zoom; touch: one finger scrolls the page (`touch-action: pan-y`), a tap picks, two fingers pan and pinch. Keyboard: the plot is focusable — arrows move a crosshair (the view follows), +/− zoom, 0 resets, Enter calls `onPick`; the position is announced. The wrapper is `role="group"`; the canvases are one `role="img"` with `ariaLabel`; zoom buttons are a toolbar outside the image. Minimizers are + crosses labelled "𝐱⋆ = (1, 1)" (`minimaLabels`, default when there is one); `start` draws a shared 𝐱₀ ring. **`overlays: Overlay2D[]`** (declarative, data coordinates, see `overlays2d.ts`): `polygon` (fill/label: simplex, polytope), `polyline`, `segment`, `arrow` (labelled: −∇f, pₖ), `disk` (trust region), `ellipse` ({x : (x−c)ᵀM(x−c) = r²} from a 2×2 SPD matrix), `implicit` (g(x) = c by marching squares over the view, cached), `constraints` (feasible-set shading: the infeasible side hatched + tinted, boundaries gᵢ = 0 and hⱼ = 0 drawn), `region` (hatch where `inside(x, y)`), `point` (dot/ring/cross + label), `text`. `overlay(ctx, view)` still draws anything else with `view.toPx` (under the paths); `overlayAfter(ctx, view)` draws after the paths (labels that must stay on top). `axisLabels` also name the hover readout and the keyboard announcement; `valueLabel` (default `f`, e.g. `‖F‖`) names the value. `initialView` frames a window (and is the Reset target) while levels and colors still come from `domain`. `onFieldReady()` fires once per field when its raster is first shown (autoplay can wait for the landscape). |
| `drawPathLayer(ctx, paths, {t, toPx, halo, ease, bounds, offViewLabels})` / `PathSpec`                                                                              | Used by Contour2D; reuse on any 2-D canvas. Eased interpolation between steps, fading trail, halo, glowing head, optional arrows/dots, `muted`, `dash`, `width`, `milestones` (diamonds at k = 10ⁿ), `end` (solid/hollow end marker), `start: false`, `quiet`. With `bounds`, steps that leave the view are dashed with chevrons at the exit and re-entry and labelled "𝐱₂ = (0.76, −3.18) / Newton, below the view". `overlappingPaths(paths, tol)` → `[i, j]` pairs where path i runs along path j (≥ 90 % of its arc length within `tol`): draw i on top, dashed, and say so; `pathCoverage(a, b, tol)` is the underlying measure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `<ConvergenceChart series t yLabel yName logY logX xDomain yDomain slopes onSeek onHover compact>`                                                                  | Error vs k (or vs any x: `series.x`, e.g. h), one series per method, faint future, playhead, hover tooltip, legend (with `count`), empty state. `logX` = log k (k ≥ 1, labelled "iteration k ≥ 1"; `suggestLogK(lengths)` says when: ≥ 30× spread). Milestone diamonds at k = 10, 10², …; `end: 'converged' \| 'stopped'` (solid / hollow end dot); `rate: 'quadratic'` (Newsreader italic note, only when proven); `slopes: [{ slope: 2, at }]` draws slope triangles on log-log axes. Powers of ten are typeset in the KaTeX fonts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `<IterationTable steps k columns onSelect>`                                                                                                                         | Virtualized; current row highlighted and followed; `defaultColumns` (k, x, f, ‖∇f‖, α) or your own `Column[]`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `<MatrixView matrix previous pivot rows cols cells separatorBefore rowLabels colLabels>`                                                                            | Elimination steps / simplex tableaux; values cross-fade on change, cells that differ from `previous` get a fading wash, pivot ring (iris), band shading, KaTeX labels allowed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `<TableView columns rows highlight futureAfter onSelect caption maxHeight>`                                                                                         | Semantic `<table>` with KaTeX headers (`{ key, tex: '\\                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | \\nabla f\\ | ', value }`), numbers right-aligned in tabular mono, current row = soft iris + 2 px rule, scrolled into view. For thousands of rows use `IterationTable`. |
| `<TreeView nodes k current onSelect>` / `layoutTree`                                                                                                                | Search trees (branch and bound): `{ id, parent, label, detail, edge, status, step }`; tidy layout, revealed up to step `k`, statuses `open · branched · pruned · infeasible · incumbent · optimal` drawn with a word and a shape (never color alone), current node ringed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `<DataPlot points onPointsChange curves residualsTo highlight weights>`                                                                                             | Scatter + curves (fits, interpolants). With `onPointsChange` every point is a real button: drag it, or Tab to it and use the arrow keys (Shift ×10); axes freeze while dragging. Residual segments to a curve; `weights` fade down-weighted points (robust fits).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |             |
| `drawMath(ctx, runs, x, y, opts)`, `mathVar mathBold mathSub mathSup pow10Runs iterateRuns starRuns`                                                                | Math on canvases in the KaTeX fonts (variables italic, vectors bold, U+2212, 10ⁿ), so chart labels match the formulas beside them. Axis names: `xName`/`yName` runs; a one-letter `xLabel` is set as a variable automatically.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `<LazySurface3D f domain paths t resolution>`                                                                                                                       | three.js in its own chunk — always inside `<Suspense>`. Drag to orbit, wheel to zoom. Each path is one tube built once per trace/theme; playback only reveals a prefix (`setDrawRange`) and moves the head.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

### Playback (`src/play`)

`useTracePlayer(traces, {autoplay, speedIndex, loop, resetOnChange})` → `{k, t, maxK, lengths, playing,
speed, atEnd, reducedMotion, play, pause, toggle, step(±n), seek, reset, toEnd, setSpeed, faster,
slower, localT(i), localK(i)}`. `t` is continuous; method _i_ stops at its own last step
(`localK(i)`). Time-based (wall clock), ~7 s per run at 1× (2.5–90 steps/s). With
`prefers-reduced-motion` there is no autoplay and no loop: the player shows the final step, paused;
Play then steps in whole steps at ≤ `REDUCED_MOTION_RATE` (6) steps/s × speed.
`usePlayerKeyboard(player)` binds Space, ←/→ (Shift = 10), Home/End, R, +/−. It ignores typing, leaves
arrow keys to widgets that own them (radios, sliders, tabs, listboxes, popup triggers, anything with
`data-own-keys`), leaves Space to buttons, and handles Home/End only inside `[data-player-scope]` (the
LabShell stage). Pure helpers (`segmentAt`, `easeInOut`, `advance`, `baseRate`) live in `timeline.ts`.

### App

Routes: `#/` (home), `#/labs`, `#/lab/<id>` (open lab or planned-lab page), `#/methods` (`?q=`,
`family=`, `needs=`, `rate=`, `det=`), `#/method/<id>`, `#/research`, `#/research/<study id>`
(`?s=<heading slug>` scrolls to a section; `#/research/about` is `research/README.md`), `#/python`,
`#/dev`; anything else is the 404 page. `CATALOG` (`src/app/catalog.ts`) holds the
build-time counts (methods, families, problems, research notes, per family / kind) from
`virtual:numopt/catalog` — a few hundred bytes; `loadCatalogIndex()` lazily loads `virtual:numopt/index` (method,
problem and research lists) for search, the Methods/Research pages and planned labs. The raw generated JSON
never reaches the bundle; fixtures are test-only. The header has the nav (Labs · Methods · Research ·
Python) and the search palette (`/` or ⌘K / Ctrl K; a native modal `<dialog>` with the combobox pattern):
labs, methods and problems; a method opens its page (`#/method/<id>`), a problem opens the lab of its
kind (`?p=<id>`; the lab named after the kind wins, e.g. `unconstrained`, not `line-search`).

**Build-time modules** (`vite/catalog.ts`; the raw JSON and the fixtures never reach the bundle):

| Module                                      | Content                                                                                                                                                              | Loaded by                     |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `virtual:numopt/catalog`                    | counts (methods, families, problems, per family / kind)                                                                                                              | home (eager, ~300 B)          |
| `virtual:numopt/index`                      | methods (with `params`, `needs`, fixture `cases`), problems, research list                                                                                           | search, Methods, method pages |
| `virtual:numopt/parity` → `parity/<family>` | each fixture case reduced to its inputs, `nIter`, `converged`, final `x`/`fun`, message and first ten iterates (≈ 120 KB for all cases, against ≈ 10 MB of fixtures) | a method page, one family     |
| `virtual:numopt/research` → `research/<id>` | study summaries (question, finding and promotion from the `research/README.md` table) and one study's HTML, TOC and figures                                          | the Research pages            |

Research figures that a README shows (and only those) are emitted to `dist/research/<id>/figures/*`;
the dev server serves the same relative URLs, so a study works from any sub-path.

**Method page** (`#/method/<id>`): facts (id, rate, needs, randomness, parity), a **live run** of the
method's first parity case by the TS port (contour + path for 2-D objectives, curve + iterate for
scalar ones, a convergence chart for all, the shell `MethodCard` on the same player), the parameter
table from the ParamSpecs (with the port's `tex` / `label`), the Python call and CLI line with the
recorded output, the **parity replay** (every fixture case rerun in the browser and compared like
`tests/parity.test.ts`), the sources, and previous / next in the family. "Open in the lab" deep-links
`#/lab/<lab>?m=<id>~0&p=<problem>`.

`useUrlState(key, default, codec)` stores view state in the hash query (`#/lab/x?p=…&m=…`); defaults
are omitted so links stay short; `codecs.string/number/bool/numbers/strings/json()`,
`codecs.tuple(n)` (exactly n finite numbers, e.g. a start point) and `codecs.oneOf([...])` (enums).
Anything a codec rejects falls back to the default; define codecs as module constants.
`clearUrlKeys([...])` drops stale keys (e.g. `x0` when the problem changes). The app wraps everything in
`<MotionConfig reducedMotion="user">`, moves focus to the new page's `<h1>` after a client-side route
change, and loads `#/dev` lazily (dev server only). Every lazy route (labs, method pages, studies)
has its own error boundary (`RouteBoundary`): a page that throws shows a recoverable error panel, not
a blank app.

### Lab shell (`src/labs/_shell`)

| API                                                                                                                                                                        | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LabShell`, `RailSection`, `Insight {id, title, label, content, actions, grow, wide, height}`                                                                              | Layout above. `label` names an insight whose `title` is not a string (e.g. a tab list).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `ProblemPicker`, `MethodSlots {hiddenParams, allowDuplicates}`, `StartPointFields`, `MethodCard {…, error, quantities, iterationNoun}`, `RunSummary {runs, iterationNoun}` | `MethodCard` is a focusable scroll region; it shows the run status in plain language and an input error as an error, not as divergence. `quantities` replaces `doc.quantities` (an entry with `key: 'x'` replaces the 𝐱ₖ cell; `digits` sets significant digits); array values of 2+ entries get a full-width cell. `iterationNoun` (`['trial', 'trials']`) words the status. `MethodSlots.hiddenParams` is one list or a per-method record (`{ exact_quadratic: ['c1'] }`); `allowDuplicates` lets a method appear twice (keyed by color slot). `RunSummary` wraps onto a second row and hides when the stage is narrower than 560 px. |
| `useLabRuns(problem, selections, options, {defer})` → `LabRun[]`                                                                                                           | Memoized runs; with `{defer: true}` the first runs are computed at once and later ones in a task after the render (the previous runs stay, `pending` is true; a playing player cannot starve the computation, and a burst of changes costs one run). `LabRun.error` is set when a method threw (e.g. a line search's `ValueError` on a problem unbounded below).                                                                                                                                                                                                                                                                        |
| `useMethodSelection(available, default, key = 'm', {allowDuplicates})` → `[selection, set]`                                                                                | URL-backed selection, validated with `sanitizeSelection`; a toast names methods a shared link asked for that the lab does not have.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `sanitizeSelection(sel, available, {allowDuplicates})` → `{value, dropped}`, `coerceParam(spec, v)`                                                                        | Drops unknown (and, unless `allowDuplicates`, duplicate) ids, keeps ≤ 4 in the given order, gives colliding or invalid color slots a free slot, keeps declared params only (coerced and clamped).                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `selectionCodec`, `firstFreeSlot(sel)`                                                                                                                                     | `gradient_descent~0~alpha=0.002,heavy_ball~1` (id ~ slot ~ params).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `describeResult(result, error?, noun?)` → `{tone, short, long, icon}`, `divergenceReason(message)`                                                                         | "Converged in 204 iterations", "Stopped at the 1,000-iteration budget without converging", "Diverged after 12 iterations (\|x\| grew past the divergence bound)" / "(non-finite value)", "cycled", "stalled". `LabPreset.note` is a ReactNode; a string note may contain inline TeX between `$…$`.                                                                                                                                                                                                                                                                                                                                      |

Lab registry (`src/labs/index.ts`, discovered — see "How to build a lab"): `LABS: LabEntry[]` (`LabMeta` +
`status`, lazy `component`, `preload()`, `methodCount`), `LAB_GROUPS`, `getLab(id)`, `labsInGroup(group)`,
`labForFamily(family)`, `labForProblemKind(kind)`.

## Porting a method

```ts
// src/methods/roots/bracketing.ts  (mirrors src/numopt/roots/bracketing.py)
import { registerMethod, param } from '../../core/registry';
import type { MethodFn, Problem, Result, Step } from '../../core/types';

const bisection: MethodFn<Problem<number>> = (problem, { bracket, xtol, max_iter }) => { … return result; };

registerMethod({ id: 'bisection', family: 'roots', name: 'Bisection',
  params: [param.float('xtol', 1e-10, { min: 1e-15, max: 1e-2, log: true }), param.int('max_iter', 100, { min: 1, max: 10_000 })],
  needs: ['f'], order: 'linear', summary: '…', references: ['…'] }, bisection,
  { rule: 'c_k = \\tfrac{a_k + b_k}{2}', intuition: '…', quantities: [{ tex: '[a_k, b_k]', key: 'info.bracket' }] });
```

Rules: same id, family, param names/defaults/ranges as the Python `@register`; one `Step` for k = 0 and
one per iteration; the same stopping test and `converged`/`message` semantics; the same `info` keys;
count evaluations like `Counted`; never throw on numerical breakdown (return `converged: false`);
stochastic methods draw only from `new Rng(seed)` in the Python order. Keep floating-point operations
in the Python order where it matters for the first 10 iterates (e.g. `x + alpha * p`, not
`alpha * p + x` with fused rearrangements).

## Parity tests

`tests/parity.test.ts` reads every `src/generated/fixtures/<family>.json`, finds the registered TS
method and TS problem with the fixture's ids, runs it with the fixture's params (merged over the
spec defaults) and checks: first `min(10, n)` iterates within 1e-8; for deterministic methods
`nIter` exactly, final `x` within 1e-6 relative, same `converged`; for stochastic methods the final
objective within 1e-6 relative. Cases whose method or problem is not ported are **skipped and
counted**; the run prints a summary such as
`[parity] 12 fixture files, 142 cases: 30 checked, 112 skipped (method not ported)` followed by the
most common missing ids. Regenerate fixtures with `npm run gen` whenever Python changes.

RNG fixture (`tests/fixtures/rng_python.json`, 1,400+ values per the streams in
`tests/fixtures/README.md`) comes from `numopt.core.rng.Rng`. Regenerate it from the repo root with
`.venv/bin/python web/tests/fixtures/gen_rng_fixture.py`.

## Conventions

**Color.** Methods take color _slots_ 1–4 (`--series-1..4`, `seriesVar(slot)`, `colors.series[slot]`)
assigned in fixed order; a method keeps its slot while selected (color follows the method, never
its rank; `firstFreeSlot`). The four slots (blue, orange, aqua, plum) were validated with the dataviz
palette validator _all-pairs_ in both themes (CVD ΔE ≥ 9.2, normal-vision ΔE ≥ 18.8), because paths
can cross. Never add a 5th series color; never use status colors (`--color-good/warn/bad`) for data.
A colored mark is always accompanied by the method name (chip, legend, table) — identity is never
color alone. Text stays in text tokens, never in series colors. Field colormaps
(`CONTOUR_MAPS.light/dark`) sit in a lightness band paths never use, so paths stay legible over any
band, and their basin is a low-chroma teal-gray, not blue (slot 1 is blue and is always the first
method); `SEQUENTIAL` (cividis-like) is for heatmaps/3-D where the field is the content. Never use a
rainbow map. Raw hex values belong only in `tokens.css` / `colors.ts`.

**Type & space.** Inter Variable for UI, JetBrains Mono for numbers in tables and ticks (tabular);
4-px spacing scale (`--space-*`), radii `--radius-*`, type scale `--text-*`. Use CSS modules + tokens;
no inline magic numbers beyond layout one-offs.

**Motion.** Motion explains state changes; it is never decoration. Use `--duration-*`/`--ease-*`
tokens in CSS and the `motion` library for layout/enter/exit. Every animation must respect
`prefers-reduced-motion` (the tokens collapse to 0 ms; `MotionConfig reducedMotion="user"` covers the
motion library; the player shows the final step and does not autoplay or loop; looping hero animations
have a visible pause button; `usePrefersReducedMotion()` for JS).

**Accessibility.** Every control is reachable and operable by keyboard with a visible, opaque focus
ring (`--color-focus`, ≥ 3:1); text tokens reach 4.5:1 on every surface (`--color-text-3` included);
icon-only buttons have `label`s; canvases have `role="img"` + a descriptive `aria-label` (never around
interactive children) and the same data is available as text (IterationTable, MethodCard). Components
that use arrow keys `stopPropagation`, and the player skips widgets with arrow-key roles. Touch targets
are ≥ 24 × 24 px. Verify with axe (0 violations on `#/` and `#/lab/<id>` in both themes).

**Performance.** Methods run synchronously in `useLabRuns` (memoized; `{defer: true}` for heavy
families); keep demo sizes modest. Canvas components redraw at most once per frame; heavy fields go to a
worker (see `Contour2D`); three.js is only loaded through `LazySurface3D`, KaTeX only through
`Formula`/`preloadKatex`, and a lab's methods only through its `setup` (the home page stays at ~540 KB).

**Lint rules.** `eslint-plugin-react-hooks` v7 enforces React Compiler rules: no ref reads/writes
during render (sync refs in `useLayoutEffect`), no synchronous `setState` in effects (adjust state
during render with a "previous value" state instead), and component files export components only.

## Home page — approved design (do not revert)

The repo owner approved this design ("ideal and clean; it should not throw off the reader"). Keep
these decisions; propose changes to the owner before you make them.

1. **Header and footer** stay as they are (logo, Labs · Methods · Research · Python, search, theme).
2. **Hero** on the 12-column grid: text in 5 columns, figure in 7 (stacked below 1080 px, the figure
   below the text). The Newsreader headline "Numerical optimization, _iterate by iterate._", one
   lede sentence, two CTAs ("Open a lab" scrolls to the gallery, "Browse methods" opens
   `#/methods`), and one quiet line of facts from `CATALOG`. No eyebrow pill, no code card, no
   facts strip, no syllabus text index, no MethodCard showcase.
3. **Hero figure** (`src/app/home/HeroShowcase.tsx`): a live showcase that cycles through
   `HERO_SCENES` (`src/app/home/heroScenes.ts`: descent race, root finding, simplex walk, Runge vs
   Chebyshev). Each scene plays once, holds its final frame, then crossfades to the next. Under the
   figure: the method legend, the scene switcher with a progress bar, a pause button, and a caption
   that states the start, the stopping test and the counts, with "Open this lab →". The figure pauses
   when it scrolls out of view. Reduced motion: no cycling, no progress bar, final frames only.
   Every scene must show runs that converge (or say plainly why not) — no budget-stopped runs in the
   first impression.
4. **Labs gallery** (`src/app/LabsIndex.tsx` + `src/app/home/LabCard.tsx`) is the main body: one
   card per lab, grouped by `LAB_GROUPS`, 3 / 2 / 1 columns (≥ 1024 / ≥ 640 / phones). A card is a
   16 : 10 thumbnail, the title, the pitch from `meta.ts` and the method count from the registry;
   the whole card is one link. The thumbnail is the end frame of the lab's own preview; hover or
   keyboard focus plays it once (≤ 3.2 s); reduced motion keeps the still. `#/labs` uses the same
   gallery.
5. **Previews** (`src/app/home/previews/labs/<lab id>.ts`, one lazy chunk each, loaded when a card
   nears the viewport): each runs the real TS ports on a registered problem when it loads and draws
   from the traces (no fixtures, nothing hand-placed). A preview exports `build(): Preview` with
   `title`, `caption` (provenance), `legend` (≤ 4 methods, color slots), `ariaLabel`, `duration` and
   `draw(ctx, size, colors, u, hero)` for progress u ∈ [0, 1] (u = 1 is the poster). Shared marks and
   the cached contour field are in `previews/draw.ts`. A new lab needs a preview:
   `previews.test.ts` fails when a lab folder has none.
6. Calm neutrals, one accent, generous whitespace, both themes, no horizontal overflow at 390 px.
   The Python code snippet (`CodeCard`) lives on `#/python` only.

## Demo code

The temporary demo ports (`src/labs/unconstrained/_demo.ts`) are gone: every lab, the home previews
and `#/dev` run the real ports, and every lab's methods are checked against the Python fixtures by
`npm test` (226 cases, all ported).

## Visual verification

Build, `npx vite preview --port <your port>`, and screenshot `#/`, `#/lab/<id>`, `#/methods`, `#/method/<id>`,
`#/research/<id>` and `#/dev` with Playwright
at 1440×900 and 390×844 in both color schemes (`colorScheme: 'dark'`); also check
`document.documentElement.scrollWidth <= innerWidth` (no horizontal overflow).
