"""Render the one-page brand sheet (docs/brand/brand-sheet.png) from the same values brand.md quotes.

    .venv/bin/python docs/brand/scripts/make_sheet.py

The sheet is an HTML page set in the real fonts (vendored woff in scripts/fonts, KaTeX from
web/node_modules) and screenshotted with headless Chromium. Colors come from palette_check.THEMES
and make_hero.colormap, so the sheet cannot disagree with the tables in brand.md.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import make_hero as hero
from colorlib import contrast, rgb_to_hex
from palette_check import THEMES

HERE = Path(__file__).resolve().parent
BRAND = HERE.parent
REPO = HERE.parents[2]
KATEX = REPO / "web" / "node_modules" / "katex" / "dist"

SEQ_ANCHORS = [
    (0, 0.27, 0.09, 268),
    (0.35, 0.47, 0.07, 245),
    (0.6, 0.62, 0.04, 140),
    (0.82, 0.78, 0.11, 96),
    (1, 0.93, 0.15, 98),
]


def seq(t: float) -> str:
    a = SEQ_ANCHORS
    i = 0
    while i < len(a) - 2 and t > a[i + 1][0]:
        i += 1
    t0, L0, C0, h0 = a[i]
    t1, L1, C1, h1 = a[i + 1]
    u = (t - t0) / (t1 - t0)
    a0, b0 = C0 * math.cos(math.radians(h0)), C0 * math.sin(math.radians(h0))
    a1, b1 = C1 * math.cos(math.radians(h1)), C1 * math.sin(math.radians(h1))
    return hero._oklab_to_hex(L0 + (L1 - L0) * u, a0 + (a1 - a0) * u, b0 + (b1 - b0) * u)


def gradient(fn, n: int = 24) -> str:
    return (
        "linear-gradient(90deg,"
        + ",".join(f"{fn(i / (n - 1))} {100 * i / (n - 1):.1f}%" for i in range(n))
        + ")"
    )


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(v: float, digits: int = 2) -> str:
    """1.22×10⁻¹⁶ with a true minus sign (the portal's `sci` in web/src/core/format.ts)."""
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    m = v / 10**e
    return f"{m:.{digits}f}×10{str(e).translate(SUP)}".replace("-", "−")


def bfgs_rows() -> str:
    """The last four BFGS iterates on Rosenbrock, from the real trace."""
    import numpy as np

    import numopt
    from numopt import problems

    p = problems.get("rosenbrock")
    r = numopt.run("bfgs", p, x0=[-1.2, 1.0])
    rows = []
    for st in r.trace[-4:]:
        cls = ' class="cur"' if st.k == r.trace[-2].k else ""
        a = st.step_size if st.step_size is not None else float("nan")
        rows.append(
            f"<tr{cls}><td>{st.k}</td><td>{st.x[0]:.6f}</td><td>{st.x[1]:.6f}</td>"
            f"<td>{sci(st.fun)}</td><td>{sci(float(np.linalg.norm(p.grad(np.asarray(st.x)))), 1)}</td><td>{a:.3f}</td></tr>"
        )
    return "".join(rows)


def chip(hexv: str, name: str, on: str, ink: str) -> str:
    c = contrast(hexv, on)
    return (
        f'<div class="chip"><div class="sw" style="background:{hexv}"></div>'
        f'<div class="cn" style="color:{ink}">{name}</div><div class="cv">{hexv} · {c:.2f}:1</div></div>'
    )


def build() -> str:
    L, D = THEMES["light"], THEMES["dark"]
    font = lambda f: (HERE / "fonts" / f).as_uri()  # noqa: E731
    method_names = [m.short for m in hero.METHODS]
    series_l = list(L["series"].values())
    series_d = list(D["series"].values())

    def method_row(series, surface, ink, field_fn):
        cells = []
        for i, (hx, nm) in enumerate(zip(series[:4], method_names, strict=False)):
            cells.append(
                f'<div class="m"><svg width="150" height="44" viewBox="0 0 150 44"><rect width="150" height="44" rx="8" fill="{field_fn(0.35)}"/>'
                f'<path d="M10 34 C 40 6, 70 40, 100 16 S 135 20, 140 12" fill="none" stroke="{surface}" stroke-opacity=".85" stroke-width="5" stroke-linecap="round"/>'
                f'<path d="M10 34 C 40 6, 70 40, 100 16 S 135 20, 140 12" fill="none" stroke="{hx}" stroke-width="2" stroke-linecap="round"/></svg>'
                f'<div class="mn" style="color:{ink}"><i style="background:{hx}"></i>{i + 1} · {nm}</div><div class="cv">{hx} · {contrast(hx, surface):.2f}:1</div></div>'
            )
        return "".join(cells)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{(KATEX / "katex.min.css").as_uri()}">
<script src="{(KATEX / "katex.min.js").as_uri()}"></script>
<style>
@font-face{{font-family:Newsreader;src:url({font("newsreader-latin-400-normal.woff")});font-weight:400}}
@font-face{{font-family:Newsreader;src:url({font("newsreader-latin-500-normal.woff")});font-weight:500}}
@font-face{{font-family:Newsreader;src:url({font("newsreader-latin-400-italic.woff")});font-style:italic}}
@font-face{{font-family:Inter;src:url({font("inter-latin-400-normal.woff")});font-weight:400}}
@font-face{{font-family:Inter;src:url({font("inter-latin-500-normal.woff")});font-weight:500}}
@font-face{{font-family:Inter;src:url({font("inter-latin-600-normal.woff")});font-weight:600}}
@font-face{{font-family:JBM;src:url({font("jetbrains-mono-latin-400-normal.woff")})}}
*{{box-sizing:border-box;margin:0}}
body{{width:1600px;background:{L["bg"]};font:15px/1.55 Inter;color:{L["text"]};font-feature-settings:'tnum'}}
.wrap{{padding:72px 80px}}
h2{{font:600 12px Inter;letter-spacing:.08em;text-transform:uppercase;color:{L["text-3"]};margin:0 0 20px}}
section{{padding:44px 0;border-top:1px solid rgba(20,20,19,.085)}}
.top{{display:flex;justify-content:space-between;align-items:flex-end;padding-bottom:48px}}
.tag{{font:400 64px/1.04 Newsreader;letter-spacing:-.022em}}
.tag span{{color:{L["text-2"]}}}
.cols{{display:grid;grid-template-columns:repeat(3,1fr);gap:40px}}
.p h3{{font:500 26px/1.2 Newsreader;letter-spacing:-.01em;margin-bottom:8px}}
.p p{{color:{L["text-2"]};font-size:15px}}
.type{{display:grid;grid-template-columns:1.15fr 1fr;gap:56px;align-items:start}}
.spec .k{{font:500 12px Inter;color:{L["text-3"]};margin:22px 0 6px;letter-spacing:.02em}}
.d1{{font:400 56px/1.05 Newsreader;letter-spacing:-.022em}}
.d2{{font:500 30px/1.2 Newsreader;letter-spacing:-.012em}}
.t1{{font:400 17px/1.6 Inter;color:{L["text-2"]};max-width:620px}}
.ui{{font:500 14px Inter}}
.mono{{font:400 13.5px/1.7 JBM;color:{L["text-2"]}}}
.mathbox{{background:{L["surface"]};border:1px solid rgba(20,20,19,.085);border-radius:14px;padding:24px 28px;font-size:21px}}
table.it{{border-collapse:collapse;width:100%;font:400 13px/1 JBM;margin-top:14px}}
table.it th{{font:500 12px Inter;color:{L["text-3"]};text-align:right;padding:8px 10px;border-bottom:1px solid rgba(20,20,19,.12)}}
table.it td{{text-align:right;padding:8px 10px;border-bottom:1px solid rgba(20,20,19,.06);color:{L["text-2"]}}}
table.it tr.cur td{{background:rgba(67,72,212,.08);color:{L["text"]}}}
.chips{{display:flex;gap:14px;flex-wrap:wrap}}
.chip{{width:150px}}
.sw{{height:64px;border-radius:10px;border:1px solid rgba(20,20,19,.08)}}
.cn{{font:500 13px Inter;margin-top:8px}}
.cv{{font:400 11.5px JBM;color:{L["text-3"]};margin-top:2px}}
.themes{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}
.panel{{border-radius:18px;padding:28px;border:1px solid rgba(20,20,19,.085)}}
.panel.dk{{background:{D["bg"]};border-color:rgba(255,255,250,.08)}}
.panel.dk .cv{{color:{D["text-3"]}}}
.panel.lt{{background:{L["surface"]}}}
.pt{{font:600 13px Inter;margin-bottom:16px}}
.ms{{display:flex;gap:12px;flex-wrap:wrap}}
.m .mn{{font:500 13px Inter;margin-top:8px;display:flex;align-items:center;gap:7px}}
.m .mn i{{width:9px;height:9px;border-radius:50%;display:inline-block}}
.ramp{{height:36px;border-radius:10px;margin:6px 0 4px}}
.rl{{display:flex;justify-content:space-between;font:400 12px JBM;color:{L["text-3"]}}}
.marks{{display:flex;gap:28px;align-items:center}}
.marks .b{{width:140px;height:140px;border-radius:20px;display:grid;place-items:center}}
</style></head><body><div class="wrap">
<div class="top"><img src="{(BRAND / "logo" / "wordmark-light.svg").as_uri()}" height="76">
<div class="cv" style="text-align:right">docs/brand · generated by scripts/make_sheet.py</div></div>
<div class="tag">Numerical optimization,<br><span>iterate by iterate.</span></div>

<section style="margin-top:56px"><h2>Principles</h2><div class="cols">
<div class="p"><h3>Every iterate, on the record.</h3><p>The trace is the product. Each step is drawn, tabulated and replayable; nothing is smoothed, skipped or hand-placed.</p></div>
<div class="p"><h3>Cited, then drawn.</h3><p>A method earns a color only after it names its source, states its stopping test and matches its fixture in both languages.</p></div>
<div class="p"><h3>Quiet chrome, loud mathematics.</h3><p>Neutral surfaces, one accent, thin marks. Color is spent on identity and magnitude, never on decoration.</p></div>
</div></section>

<section><h2>Type</h2><div class="type"><div class="spec">
<div class="k">DISPLAY · Newsreader 400–500, opsz display, −2.2 % tracking</div><div class="d1">Watch the valley close.</div>
<div class="k">HEADING · Newsreader 500</div><div class="d2">Quasi-Newton methods</div>
<div class="k">TEXT · Inter 400, 15–17 px, 1.55–1.6 leading</div><div class="t1">BFGS builds a curvature model from successive gradients, so it needs no Hessian and still converges superlinearly near a strong minimizer.</div>
<div class="k">UI · Inter 500, 13–14 px</div><div class="ui">Step size &nbsp;·&nbsp; Gradient tolerance &nbsp;·&nbsp; Iteration budget</div>
</div><div class="spec">
<div class="k">MATH · KaTeX (Computer Modern metrics)</div>
<div class="mathbox" id="math"></div>
<div class="k">NUMBERS · JetBrains Mono, tabular, U+2212 minus, ×10ⁿ · iterates bold (𝐱ₖ), coordinates x, y</div>
<table class="it"><tr><th>k</th><th id="h-x"></th><th id="h-y"></th><th id="h-f"></th><th id="h-g"></th><th id="h-a"></th></tr>
{bfgs_rows()}</table>
</div></div></section>

<section><h2>Color · neutrals and the signature accent</h2><div class="themes">
<div class="panel lt"><div class="pt">Light</div><div class="chips">
{chip(L["bg"], "bg", L["text"], L["text"])}{chip(L["surface"], "surface", L["text"], L["text"])}{chip(L["sunken"], "sunken", L["text"], L["text"])}
{chip(L["text"], "text", L["surface"], L["text"])}{chip(L["text-2"], "text-2", L["surface"], L["text"])}{chip(L["text-3"], "text-3", L["surface"], L["text"])}
{chip(L["accent: iris (proposed)"], "iris (accent)", L["surface"], L["text"])}</div></div>
<div class="panel dk"><div class="pt" style="color:{D["text"]}">Dark</div><div class="chips">
{chip(D["bg"], "bg", D["text"], D["text"])}{chip(D["surface"], "surface", D["text"], D["text"])}{chip(D["raised"], "raised", D["text"], D["text"])}
{chip(D["text"], "text", D["surface"], D["text"])}{chip(D["text-2"], "text-2", D["surface"], D["text"])}{chip(D["text-3"], "text-3", D["surface"], D["text"])}
{chip(D["accent: iris (proposed)"], "iris (accent)", D["surface"], D["text"])}</div></div>
</div><div class="cv" style="margin-top:12px">ratio = WCAG contrast against the theme surface (neutrals: against text)</div></section>

<section><h2>Color · methods (four slots, validated all pairs and against the accent; identity always also by name)</h2><div class="themes">
<div class="panel lt"><div class="pt">Light — on the contour field, with the 2-px halo</div><div class="ms">{method_row(series_l, L["surface"], L["text"], lambda t: hero.colormap("light", t))}</div></div>
<div class="panel dk"><div class="pt" style="color:{D["text"]}">Dark</div><div class="ms">{method_row(series_d, D["surface"], D["text"], lambda t: hero.colormap("dark", t))}</div></div>
</div></section>

<section><h2>Color · fields</h2><div class="themes">
<div><div class="k cv">Contour field, light (t = 0 is the minimum)</div><div class="ramp" style="background:{gradient(lambda t: hero.colormap("light", t))}"></div>
<div class="k cv" style="margin-top:14px">Contour field, dark</div><div class="ramp" style="background:{gradient(lambda t: hero.colormap("dark", t))}"></div>
<div class="rl"><span>low f — basin</span><span>high f — walls</span></div></div>
<div><div class="k cv">Sequential (heatmaps, surfaces; the field is the content)</div><div class="ramp" style="background:{gradient(seq)}"></div>
<div class="rl"><span>0</span><span>monotone OKLCH lightness, 0.27 → 0.93</span><span>1</span></div>
<div class="k cv" style="margin-top:14px">Never: rainbow, jet, hsv — hue is reserved for method identity.</div></div>
</div></section>

<section><h2>Marks</h2><div class="marks">
<div class="b" style="background:{L["surface"]};border:1px solid rgba(20,20,19,.08)"><img src="{(BRAND / "logo" / "mark-color-light.svg").as_uri()}" width="112"></div>
<div class="b" style="background:{D["bg"]}"><img src="{(BRAND / "logo" / "mark-color-dark.svg").as_uri()}" width="112"></div>
<div class="b" style="background:{L["surface"]};border:1px solid rgba(20,20,19,.08)"><img src="{(BRAND / "logo" / "mark-mono-black.svg").as_uri()}" width="112"></div>
<div class="b" style="background:transparent"><img src="{(BRAND / "logo" / "mark-tile.svg").as_uri()}" width="128"></div>
<div style="display:flex;gap:18px;align-items:center"><img src="{(BRAND / "logo" / "mark-small-light.svg").as_uri()}" width="32"><img src="{(BRAND / "logo" / "mark-small-light.svg").as_uri()}" width="16"><img src="{(BRAND / "logo" / "favicon.svg").as_uri()}" width="32"><img src="{(BRAND / "logo" / "favicon.svg").as_uri()}" width="16">
<span class="cv">16-unit pixel master and favicon at 32 and 16 px</span></div>
</div></section>
</div>
<script>
katex.render(String.raw`\\begin{{aligned}} \\mathbf{{x}}_{{k+1}} &= \\mathbf{{x}}_k - \\alpha_k H_k \\nabla f(\\mathbf{{x}}_k) \\\\ H_{{k+1}} &= (I - \\rho_k \\mathbf{{s}}_k \\mathbf{{y}}_k^{{\\top}})\\, H_k \\,(I - \\rho_k \\mathbf{{y}}_k \\mathbf{{s}}_k^{{\\top}}) + \\rho_k \\mathbf{{s}}_k \\mathbf{{s}}_k^{{\\top}} \\end{{aligned}}`,
  document.getElementById('math'), {{displayMode: true}});
for (const [id, tex] of [['h-x', String.raw`x`], ['h-y', String.raw`y`], ['h-f', String.raw`f(\\mathbf{{x}}_k)`], ['h-g', String.raw`\\|\\nabla f(\\mathbf{{x}}_k)\\|_2`], ['h-a', String.raw`\\alpha_k`]])
  katex.render(tex, document.getElementById(id));
</script></body></html>"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "sheet.html"
        html.write_text(build())
        out = BRAND / "brand-sheet.png"
        subprocess.run(
            ["node", str(HERE / "render.mjs"), str(html), str(out), "1600", "2620", "--scale", "1"],
            check=True,
        )
        print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    _ = rgb_to_hex
    main()
