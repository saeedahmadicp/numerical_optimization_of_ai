# README design

The full draft is [`README.draft.md`](README.draft.md) (written for the repository root; it is not
installed there yet). This note explains its structure and the GitHub constraints behind it, so later
edits keep the same quality.

## Who reads it, and in what order

A visitor decides in about 10–20 seconds. A mathematician or researcher looks for four things, in
this order: *can I trust it* (status, license, citation), *is it correct* (the figure, the notation,
the stopping tests), *what is in it* (methods, families), *how do I run it* (three lines). The page
answers in that order and puts everything else below the fold.

| # | Section | Job | Length budget |
|---|---|---|---|
| 1 | Wordmark + tagline | Identity; the one-line promise | 2 lines |
| 2 | One quiet badge row | Trust signals researchers scan for first: Python ≥ 3.11, MIT, *Cite* (CITATION.cff); tests and DOI when they exist. Flat, in the brand's neutrals (`#52514e` / `#6b6963`), at most five | 1 line |
| 3 | Link row | Labs, quick start, methods, research, architecture, cite | 1 line |
| 4 | **Hero figure** (animated SVG; wide and stacked; light and dark) + caption | Proof by demonstration: four real methods under one stopping test, their real iteration counts, linear vs superlinear vs quadratic convergence visible as *shapes* | 1 screen |
| 5 | **Open the interactive labs →** | The most impressive part gets one prominent line right under the figure | 1 line |
| 6 | One-liner + one paragraph | "168 numerical methods, each cited to its algorithm and equation, tested against an oracle, and replayable iterate by iterate in the browser." | ≤ 80 words |
| 7 | Quick start | Install + four lines of Python + a compact, fair comparison with its real output | < 20 seconds to read |
| 8 | What's inside | Family table with live counts and named examples; the `Result`/`Step` contract in one paragraph | 1 table |
| 9 | Gallery | Four lab screenshots (placeholders until the labs ship) | 2 × 2 grid |
| 10 | Design principles | Cited · checked · honest about failure · one implementation in two languages · readable | 5 bullets |
| 11 | Research | One row per note in `research/`: title, one-line finding, link (generated) | grows |
| 12 | Repository map, citing, license | Where things live; BibTeX and CITATION.cff | 1 table + 1 block |

What is deliberately **absent**: a badge wall (one quiet row of four facts is the limit), a
feature checklist with emoji, a "Why numopt?" marketing section, a table of contents (GitHub renders
one from the headings), and installation alternatives above the fold.

## Package name

The PyPI name `numopt` already belongs to another project (numopt 0.0.7, an engineering-design
package whose import name is `NumOpt`). A visitor who copies `pip install numopt` would install the
wrong package. Checked on 2026-10-05 through `https://pypi.org/pypi/<name>/json`: `numopt-lab`,
`numoptlab`, `numopt-ref`, `numopt-reference` and `pynumopt` are free.

* **Recommendation:** publish as **`numopt-lab`** (it matches "interactive labs") and keep
  `import numopt`. The other project's module is `NumOpt`; Python imports are case-sensitive, so the
  only clash is a user who installs *both* on a case-insensitive file system (macOS, Windows), where
  the two directories merge. That risk is small; note it in the package's FAQ.
* **Until the name is reserved,** every install line (README, portal concept) is the name-free
  `pip install git+https://github.com/saeedahmadicp/numopt`. A
  `numopt-lab @ git+https://...` line would fail while `pyproject.toml` still says `name = "numopt"`.
* The rename itself is a one-line change to `[project] name` in `pyproject.toml`, owned by the
  package team.

## The hero

`scripts/make_hero.py` computes everything: it runs `gradient_descent`, `momentum`, `bfgs` and
`pure_newton` from 𝐱₀ = (−1.2, 1) on Rosenbrock and draws the actual iterates. Changing a method, a
parameter or the start point regenerates a correct figure; nothing is hand-placed. Choices:

* **One stopping test for all four methods:** the first k with ‖∇f(𝐱ₖ)‖₂ ≤ 10⁻⁸, inside one
  20,000-iteration budget; every other parameter is the registered default. The registered defaults
  differ (gradient descent and momentum stop at ‖∇f‖₂ ≤ 10⁻⁶; BFGS and Newton at ‖∇f‖∞ ≤ 10⁻⁸), so
  comparing them as registered would compare different tests. BFGS and Newton run with their
  internal tolerance at its floor and the trace is cut at the first iterate that passes the common
  test; the methods are deterministic, so this *is* the run with that test. Results: Newton 6,
  BFGS 38, heavy-ball momentum 4,128, gradient descent (Armijo backtracking) 15,231 — all converge.
* **Four methods, four validated colors.** Adam was dropped: it told the same "slow first-order"
  story as momentum, and a fifth method needed a fifth, unvalidated color (the withdrawn "reference
  ink"). With four, the lab can replay the README figure exactly.
* **Legibility at real size.** The wide figure is laid out at 960 × 650 units, so at the 830 px
  README column 1 unit ≈ 0.86 px and the smallest type (14 units) is 12 px. On a phone the
  `<picture>` serves a stacked 540 × 1206 variant (17-unit minimum type ≈ 11 px at 360 px). The
  provenance text is no longer inside the image; the caption carries it.
* **Nothing hidden.** Gradient descent, the slowest method, is painted last with the thinnest stroke,
  so it is visible along the whole valley floor. Diamonds mark 𝐱ₖ at k = 10, 10², 10³, 10⁴ on both
  panels, which ties each method's progress along the valley to the log-k axis.
* **Equal scale, named axes.** The landscape uses the same scale on x and y (125 units per unit),
  with *x* and *y* named and ticks outside the frame. The log-k axis says "iteration k ≥ 1", since
  it cannot show k = 0.
* **Newton's excursion is shown, not cropped away.** Its second iterate is 𝐱₂ = (0.76, −3.18), far
  below the view. Both steps through the frame edge are dashed in every frame, an outward chevron
  marks where 𝐱₁ → 𝐱₂ leaves, an inward (upward) chevron marks where 𝐱₂ → 𝐱₃ comes back, and the
  label sits at the exit point.
* **Two panels** — the landscape (where the iterates went) and f(𝐱ₖ) − f⋆ against k on log–log axes
  (how fast). Quadratic and superlinear convergence appear as different curve shapes, annotated in
  italics.
* **Animation on one clock that is linear in log k**, which gives each decade of iterations the same
  screen time, so the convergence playhead moves at constant speed. It **plays once** (about 9.5 s)
  and holds a final frame identical to the static figure (brand.md §7 bans endless loops); the
  heads and playhead leave when the clock stops.
* **Reduced motion:** the animated SVG contains the static final frame and a
  `@media (prefers-reduced-motion: reduce)` rule that shows it instead of the animation (confirmed
  inside `<img>` with Chromium's `--force-prefers-reduced-motion`).
* **Accessible text** (`<desc>`, alt) uses the brand's typography: U+2212, thousands separators,
  x₀, 10⁻⁸.

| File | Size | Use |
|---|---|---|
| `readme-hero/hero-animated-{light,dark}.svg` | ~345 kB | README default, desktop (SMIL; renders in GitHub's `<img>`) |
| `readme-hero/hero-stacked-animated-{light,dark}.svg` | ~350 kB | README on viewports ≤ 640 px |
| `readme-hero/hero-{light,dark}.svg`, `hero-stacked-{light,dark}.svg` | ~225 kB | Static figure: docs, slides, portal concept |
| `build/readme-hero/*.png`, `*.webp` | — | Raster fallbacks (PyPI, social posts), built by `make_hero.py`, **git-ignored** |

The raster fallbacks (formerly 4.5 MB of WebP and 1.2 MB of PNG in the tree) are not committed: they
would stay in git history forever and the README does not use them. Attach them to a release or
publish them with GitHub Pages when a host needs them.

## The compact comparison

The CLI's `numopt compare` output is about 200 characters wide, mixes ‖∇f(x)‖ and ‖∇f‖∞, nests
parentheses and says `max_iter=5000`. The README therefore shows a five-line Python loop that prints
method · iterations · f(x) − f⋆ · stop, with its real output, and states which norm each method's
`gtol` tests. **Request to the package team** (outside `docs/brand/`): add `numopt compare
--compact` with those four columns, and one message format for every method — "‖∇f‖₂ = 9.98×10⁻⁹
≤ 10⁻⁸", "Stopped at the 5,000-iteration budget · ‖∇f‖₂ = 1.17×10⁻³ > 10⁻⁶" — so the README can show
the CLI again.

## GitHub rendering constraints (and how the assets meet them)

* **README SVGs render through `<img>`:** no scripts, no external fonts, no external CSS. Every glyph in
  the hero, wordmark and social card is therefore converted to outlines with fontTools
  (`scripts/typeset.py`), using Inter, Newsreader and the KaTeX fonts (including KaTeX Main Bold for
  the bold vectors) — the figure's math is real Computer Modern, identical on every OS.
* **SMIL and CSS animations inside an SVG do run in `<img>`**, and media queries inside the SVG
  (`prefers-reduced-motion`) are honored. Keep each animated file under ~1 MB so it is not lazily
  replaced; ours are ~350 kB.
* **Theme and width switching** use one `<picture>` with four `<source>` elements, most specific
  first: `(max-width: 640px) and (prefers-color-scheme: dark)`, `(max-width: 640px)`,
  `(prefers-color-scheme: dark)`, then the `<img>` fallback (wide, light). Do not use the
  `#gh-dark-mode-only` fragment trick.
* **Relative paths** (`docs/brand/...`) so forks and branches render their own assets. GitHub's
  rewriting of *relative* `srcset` paths inside `<source>` must be verified on github.com in both the
  repository home view and the file (blob) view before merging (see the checklist below).
* **Alt text states the result** (who converged in how many iterations, under which test), not
  "hero image".
* **Badges** come from shields.io with `style=flat-square`, `labelColor=52514e` and color `6b6963`,
  so they read as quiet metadata, not as a second color system.
* **Social preview:** upload `docs/brand/logo/social-preview.png` (1280 × 640) in Settings → General →
  Social preview; GitHub does not read it from the repository. The card says "150+ methods", because
  it is not rebuilt on every release.

## Before publishing the draft

1. Run `bash docs/brand/scripts/build.sh`. It rebuilds every asset and prints the "What's inside"
   and "Research" tables from the registry and from `research/`. Then run
   `.venv/bin/python docs/brand/scripts/readme_facts.py --write docs/brand/README.draft.md`: it
   replaces both tables and the counts in the one-liner (168) and the `numopt problems` sentence
   (103). `--check` exits with status 1 while any of them is stale.
2. Resolve every `TODO(...)` marker: portal URL, PyPI name (`numopt-lab`), tests and DOI badges,
   screenshots, citation authors.
3. Copy `docs/brand/CITATION.draft.cff` to the repository root as `CITATION.cff` with the
   maintainers' names, so GitHub shows "Cite this repository" (the *Cite* badge links to it).
4. Copy `README.draft.md` to the repository root as `README.md` (its paths already assume the root).
5. Push to a branch and check on github.com before merging: light and dark themes, a phone-width
   viewport (stacked hero), the repository home view and the blob view of `README.md` (relative
   `srcset` paths), and that the animation plays once and the reduced-motion frame appears.
