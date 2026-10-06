"""Set text and simple math as SVG outlines (fontTools), so figures render identically everywhere.

GitHub shows README SVGs through <img>, which cannot load web fonts, and system fallbacks differ
per OS. Every glyph in the brand figures is therefore an outline. Text uses Inter (UI voice) or
Newsreader (display voice); math uses the KaTeX fonts (Computer Modern metrics), so a figure label
reads exactly like the LaTeX in the lab's MethodCard.

A *run* is (text, font, size, dy): `font` is a key of FONT_FILES, `dy` shifts the baseline (negative
= up, for superscripts). `math("f(x_k) - f^*", size)` builds runs from a tiny TeX subset:
single-letter italics, digits/operators upright, `_x` / `^x` / `_{..}` / `^{..}` scripts, `\\star`,
and `\\mathbf{x}` (bold upright, KaTeX_Main-Bold) for vectors. The brand writes iterates in bold
(𝐱ₖ, 𝐱⋆) so that x and y stay free for coordinates (brand.md, "Numerals and notation").
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

logging.getLogger("fontTools").setLevel(logging.ERROR)

FONTS = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {
    "inter": "inter-latin-400-normal.woff",
    "inter-medium": "inter-latin-500-normal.woff",
    "inter-semibold": "inter-latin-600-normal.woff",
    "mono": "jetbrains-mono-latin-400-normal.woff",
    "serif": "newsreader-latin-400-normal.woff",
    "serif-medium": "newsreader-latin-500-normal.woff",
    "serif-semibold": "newsreader-latin-600-normal.woff",
    "serif-italic": "newsreader-latin-400-italic.woff",
    "math-it": "KaTeX_Math-Italic.ttf",
    "math-rm": "KaTeX_Main-Regular.ttf",
    "math-rm-it": "KaTeX_Main-Italic.ttf",
    "math-bf": "KaTeX_Main-Bold.ttf",
}

Run = tuple[str, str, float, float]


@cache
def _font(key: str) -> TTFont:
    return TTFont(str(FONTS / FONT_FILES[key]))


@cache
def _kerning(key: str) -> dict[tuple[str, str], int]:
    """Pair kerning from GPOS PairPos lookups (formats 1 and 2); empty when absent."""
    font = _font(key)
    pairs: dict[tuple[str, str], int] = {}
    if "GPOS" not in font:
        return pairs
    for lookup in font["GPOS"].table.LookupList.Lookup:
        if lookup.LookupType not in (2, 9):
            continue
        for st in lookup.SubTable:
            if lookup.LookupType == 9:
                if st.ExtensionLookupType != 2:
                    continue
                st = st.ExtSubTable
            if st.Format == 1:
                cov = st.Coverage.glyphs
                for i, ps in enumerate(st.PairSet):
                    for pv in ps.PairValueRecord:
                        v = getattr(pv.Value1, "XAdvance", 0) or 0
                        if v:
                            pairs.setdefault((cov[i], pv.SecondGlyph), v)
            elif st.Format == 2:
                c1 = st.ClassDef1.classDefs
                second: dict[int, list[str]] = {}
                for g, c in st.ClassDef2.classDefs.items():
                    second.setdefault(c, []).append(g)
                for g1 in st.Coverage.glyphs:
                    rec = st.Class1Record[c1.get(g1, 0)]
                    for c, r2 in enumerate(rec.Class2Record):
                        v = getattr(r2.Value1, "XAdvance", 0) or 0
                        if v and c != 0:
                            for g2 in second.get(c, []):
                                pairs.setdefault((g1, g2), v)
    return pairs


def _glyph(key: str, ch: str, tnum: bool) -> str | None:
    font = _font(key)
    name = font.getBestCmap().get(ord(ch))
    if name and tnum:
        alt = f"{name}.tf"
        if alt in font.getGlyphOrder():
            return alt
    return name


def measure(runs: list[Run], tnum: bool = False, tracking: float = 0.0) -> float:
    return _layout(runs, 0.0, 0.0, None, tnum, tracking)


def _ntos(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _layout(runs, x, y, out: list[str] | None, tnum, tracking) -> float:
    cursor = x
    for text, key, size, dy in runs:
        pen = SVGPathPen(_font(key).getGlyphSet(), ntos=_ntos) if out is not None else None
        font = _font(key)
        upm = font["head"].unitsPerEm
        s = size / upm
        gs = font.getGlyphSet()
        hmtx = font["hmtx"]
        kern = _kerning(key)
        prev = None
        for ch in text:
            g = _glyph(key, ch, tnum)
            if g is None:
                continue
            if prev is not None:
                cursor += kern.get((prev, g), 0) * s
            if pen is not None:
                gs[g].draw(TransformPen(pen, (s, 0, 0, -s, cursor, y + dy)))
            cursor += hmtx[g][0] * s + tracking * size
            prev = g
        if pen is not None and out is not None:
            out.append(pen.getCommands())
    return cursor - x


def path_d(
    runs: list[Run],
    x: float,
    y: float,
    *,
    anchor: str = "start",
    tnum: bool = False,
    tracking: float = 0.0,
) -> tuple[str, float]:
    """SVG path data for `runs` with the baseline at y; anchor: start | middle | end."""
    w = measure(runs, tnum, tracking)
    if anchor == "middle":
        x -= w / 2
    elif anchor == "end":
        x -= w
    out: list[str] = []
    _layout(runs, x, y, out, tnum, tracking)
    return "".join(out), w


def text(s: str, size: float, font: str = "inter") -> list[Run]:
    return [(s, font, size, 0.0)]


# ── A tiny TeX subset ──────────────────────────────────────────────────────────────────

_UPRIGHT = set("0123456789()[]{},.;:=+−-|/<>!'≈·×")
_SYMBOLS = {
    "\\star": ("⋆", "math-rm"),
    "\\approx": ("≈", "math-rm"),
    "\\cdot": ("·", "math-rm"),
    "\\infty": ("∞", "math-rm"),
    "\\le": ("≤", "math-rm"),
    "\\ge": ("≥", "math-rm"),
    "\\nabla": ("∇", "math-rm"),
    "\\|": ("‖", "math-rm"),
    "\\alpha": ("α", "math-it"),
    "\\kappa": ("κ", "math-it"),
    "\\lambda": ("λ", "math-it"),
    "\\to": ("→", "math-rm"),
    "\\times": ("×", "math-rm"),
    "\\,": (" ", "space-thin"),
    "\\;": (" ", "space-med"),
}
SCRIPT = 0.70
SUB_DY, SUP_DY = 0.22, -0.42


def math(tex: str, size: float) -> list[Run]:
    """Runs for a small TeX subset (see module docstring). Spaces are ignored, as in TeX math."""
    runs: list[Run] = []
    i = 0
    prev: list[str] = [""]  # previous atom: a minus after "(", "," or "=" is unary (no spacing)

    def atom(tok: str, sz: float, dy: float, bold: bool = False) -> None:
        if bold and tok.isalpha() and len(tok) == 1:
            runs.append((tok, "math-bf", sz, dy))
            prev[0] = tok
            return
        if tok in _SYMBOLS:
            ch, key = _SYMBOLS[tok]
            if key.startswith("space"):
                runs.append((" ", "math-rm", sz * (0.5 if key == "space-thin" else 0.8), dy))
                return
            if tok in ("\\approx", "\\le", "\\ge", "\\to") and dy == 0:
                runs.append((f" {ch} ", key, sz, dy))
                prev[0] = "="
                return
            runs.append((ch, key, sz, dy))
        elif tok in ("-", "−"):
            unary = dy != 0 or prev[0] in ("", "(", ",", "=", "[")
            runs.append(("−", "math-rm", sz, dy) if unary else (" − ", "math-rm", sz, dy))
        elif tok in ("=", "+", "≈"):
            runs.append((f" {tok} ", "math-rm", sz, dy))
            prev[0] = "="
            return
        elif tok == "*":
            runs.append(("∗", "math-rm", sz, dy))
        elif tok == ",":
            runs.append((", ", "math-rm", sz, dy))
        elif tok.isalpha() and len(tok) == 1:
            runs.append((tok, "math-it", sz, dy))
        else:
            runs.append((tok, "math-rm", sz, dy))
        prev[0] = tok

    def read_group(j: int) -> tuple[list[str], int]:
        if tex[j] == "{":
            depth, k = 1, j + 1
            while depth:
                depth += {"{": 1, "}": -1}.get(tex[k], 0)
                k += 1
            return _tokens(tex[j + 1 : k - 1]), k
        if tex[j] == "\\":
            k = j + 1
            while k < len(tex) and tex[k].isalpha():
                k += 1
            return [tex[j:k]], k
        return [tex[j]], j + 1

    toks_src = tex
    while i < len(toks_src):
        c = toks_src[i]
        if c == " ":
            i += 1
            continue
        if c in "_^":
            group, i = read_group(i + 1)
            dy = (SUB_DY if c == "_" else SUP_DY) * size
            for t in group:
                atom(t, size * SCRIPT, dy)
            continue
        if tex.startswith("\\mathbf", i):
            group, i = read_group(i + len("\\mathbf"))
            for t in group:
                atom(t, size, 0.0, bold=True)
            continue
        if c == "\\":
            k = i + 1
            if k < len(tex) and not tex[k].isalpha():
                atom(tex[i : k + 1], size, 0.0)
                i = k + 1
                continue
            while k < len(tex) and tex[k].isalpha():
                k += 1
            atom(tex[i:k], size, 0.0)
            i = k
            continue
        atom(c, size, 0.0)
        i += 1
    return runs


def _tokens(s: str) -> list[str]:
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\":
            k = i + 1
            while k < len(s) and s[k].isalpha():
                k += 1
            out.append(s[i:k])
            i = k
        elif s[i] != " ":
            out.append(s[i])
            i += 1
        else:
            i += 1
    return out
