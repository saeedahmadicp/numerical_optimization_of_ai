# Vendored font outlines

These files are read by `typeset.py` / `make_logo.py` to convert text into SVG outlines (GitHub
renders README SVGs through `<img>`, which cannot load web fonts). They are not served to users.

| File | Family | Source | License |
|---|---|---|---|
| `newsreader-latin-{400,500,600}-normal.woff`, `newsreader-latin-400-italic.woff` | Newsreader (Production Type) | `@fontsource/newsreader` 5.3.0 | SIL OFL 1.1 |
| `inter-latin-{400,500,600}-normal.woff` | Inter (Rasmus Andersson) | `@fontsource/inter` 5.3.0 | SIL OFL 1.1 |
| `jetbrains-mono-latin-400-normal.woff` | JetBrains Mono | `@fontsource/jetbrains-mono` 5.3.0 | SIL OFL 1.1 |
| `KaTeX_Main-Regular.ttf`, `KaTeX_Main-Italic.ttf`, `KaTeX_Main-Bold.ttf` (bold vectors 𝐱ₖ), `KaTeX_Math-Italic.ttf` | KaTeX fonts (Computer Modern metrics) | `katex` 0.19 (`web/node_modules/katex/dist/fonts`) | MIT (KaTeX) |
