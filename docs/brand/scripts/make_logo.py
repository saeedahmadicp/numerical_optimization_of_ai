"""Generate the numopt mark, wordmark lockups, favicon and app icon (all hand-built SVG).

The mark is a theorem, drawn exactly: steepest descent with exact line search on the quadratic
f(x) = 1/2 x^T A x, A = R diag(1, kappa) R^T, kappa = 6. From x0 = s (-kappa, -1) (eigenbasis)
every step is orthogonal to the previous one and touches the next level set tangentially, and the
iterates contract by r = (kappa - 1)/(kappa + 1) = 5/7 per step (Cauchy 1847; Nocedal & Wright
2006, Thm. 3.3). The valley is turned by -45 degrees, which makes the first step horizontal, so
EVERY step is horizontal or vertical: the zigzag becomes a staircase into the minimizer, and its
square, mitered corners show the right angle that the theorem is about. The level sets in the
mark are the ellipses through the iterates, so each segment kisses an ellipse. The path ends at
the last iterate that disappears under the x* dot; no segment is schematic.

    .venv/bin/python docs/brand/scripts/make_logo.py        # writes docs/brand/logo/*.svg + PNG icons

Two masters:
    mark_svg()        the large mark (>= 48 px): filled level sets, hollow x0 ring (the chart
                      grammar of brand.md section 9), staircase, solid x* dot
    small_mark_svg()  a 16-unit pixel master (16-47 px): no level sets, two orthogonal steps
                      with a 2-unit stroke on the pixel grid, the last iterate drawn, then a
                      separate 4-unit x* dot

Text in the wordmark is converted to outlines with fontTools, so the SVGs render identically
everywhere (GitHub serves SVG through <img>, which cannot load web fonts).
"""

from __future__ import annotations

import itertools
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from colorlib import oklch_to_hex

HERE = Path(__file__).resolve().parent
BRAND = HERE.parent
LOGO = BRAND / "logo"
FONTS = HERE / "fonts"  # woff outlines (see fonts/README.md)

# ── Brand colors (see brand.md, "Color") ─────────────────────────────────────────────
INK = "#141413"
PAPER = "#fcfcfb"
PAPER_DARK = "#f1f0eb"
BG_DARK = "#0d0d0c"
IRIS = "#4b2ace"  # signature accent on light surfaces: OKLCH 0.45 0.23 281
IRIS_DARK = "#b0a8fc"  # signature accent on dark surfaces: OKLCH 0.77 0.12 288
IRIS_HUE = 281


# Filled level sets (outermost first): equal OKLCH lightness steps on the iris hue, chroma rising
# toward the minimum, so the well deepens where the iterates converge.
def _ramp(L0: float, dL: float, C0: float, dC: float, n: int = 4) -> list[str]:
    out = []
    for i in range(n):
        h = oklch_to_hex(L0 + i * dL, C0 + i * dC, IRIS_HUE)
        assert h is not None
        out.append(h)
    return out


BANDS = {
    "color-light": _ramp(0.935, -0.04, 0.022, 0.02),
    "color-dark": _ramp(0.24, 0.05, 0.028, 0.02),
    "tile": _ramp(0.26, 0.05, 0.03, 0.02),
}

# ── Geometry of the mark ───────────────────────────────────────────────────────────────
KAPPA = 6.0  # condition number of A: the iterates contract by r = (kappa - 1)/(kappa + 1) = 5/7
THETA = math.radians(-45.0)  # valley falls to the right; makes every step axis-aligned
START = (-1.0, -1.0)  # x0 = s (START[0] kappa, START[1]) in the eigenbasis: upper left


@dataclass(frozen=True)
class Mark:
    iterates: np.ndarray  # (n, 2) in mark units, centered on the minimizer, y up
    levels: list[float]  # f values of the level sets through the iterates


def fit_scale(half: float) -> float:
    """Scale s such that the level set through x0 has a bounding-box half-width of `half`."""
    a2 = KAPPA**2 + KAPPA  # 2 f(x0) / s^2: squared semi-major axis
    b2 = a2 / KAPPA
    c, s = math.cos(THETA), math.sin(THETA)
    return half / math.sqrt(a2 * c * c + b2 * s * s)


def zigzag(n_steps: int, scale: float) -> Mark:
    """Exact-line-search steepest descent on f = 1/2 (u^2 + kappa v^2), rotated by THETA."""
    x = np.array([START[0] * KAPPA, START[1]]) * scale
    pts = [x.copy()]
    for _ in range(n_steps):
        g = np.array([x[0], KAPPA * x[1]])
        alpha = (g @ g) / (g[0] ** 2 + KAPPA * g[1] ** 2)
        x = x - alpha * g
        pts.append(x.copy())
    P = np.array(pts)
    c, s = math.cos(THETA), math.sin(THETA)
    R = np.array([[c, -s], [s, c]])
    levels = [0.5 * (p[0] ** 2 + KAPPA * p[1] ** 2) for p in P]
    Q = P @ R.T
    # Orthogonality is the theorem; with THETA = -45 degrees the steps are also axis-aligned.
    steps = np.diff(Q, axis=0)
    assert all(abs(float(a @ b)) < 1e-9 * scale**2 for a, b in itertools.pairwise(steps))
    assert np.allclose(np.min(np.abs(steps), axis=1), 0, atol=1e-9 * scale)
    return Mark(iterates=Q, levels=levels)


def ellipse_attrs(level: float, cx: float, cy: float) -> str:
    """Level set 1/2 (u^2 + kappa v^2) = level as an SVG ellipse (semi-axes sqrt(2L), sqrt(2L/kappa))."""
    a = math.sqrt(2 * level)
    b = math.sqrt(2 * level / KAPPA)
    deg = math.degrees(THETA)
    return f'cx="{cx:.3f}" cy="{cy:.3f}" rx="{a:.3f}" ry="{b:.3f}" transform="rotate({-deg:.2f} {cx:.3f} {cy:.3f})"'


def fmt_pts(P: np.ndarray, cx: float, cy: float) -> str:
    # SVG y points down; flip so the mark matches the math orientation.
    return " ".join(f"{cx + x:.3f},{cy - y:.3f}" for x, y in P)


PALETTE = {
    # variant: (level-set ink, path, x* dot, tile ground)
    "color-light": (INK, IRIS, INK, None),
    "color-dark": (PAPER_DARK, IRIS_DARK, PAPER_DARK, None),
    "mono-black": (INK, INK, INK, None),
    "mono-white": ("#ffffff", "#ffffff", "#ffffff", None),
    "tile": (PAPER, IRIS_DARK, PAPER, INK),
}


def mark_svg(*, size: int = 64, variant: str = "color-light", title: str = "numopt") -> str:
    """The large mark (use at >= 48 px) in a `size`-unit viewBox, drawn on a 64-unit master grid.

    variant: color-light | color-dark   filled iris level sets, iris path, ink x*
             mono-black | mono-white    one solid color, no tints; rings masked under the path
             tile                       app icon: ink rounded square, iris path, paper x*
    """
    if variant not in PALETTE:
        raise ValueError(variant)
    cx = cy = size / 2
    u = size / 64
    tile = variant == "tile"
    w_ring, w_path, r_dot = 1.35 * u, 2.6 * u, 4.2 * u
    r_x0, w_x0 = 3.3 * u, 1.9 * u
    # x0 sits at the left extreme of the outer level set, so the inset must also hold its ring.
    inset = 10.5 * u if tile else 1.0 * u + r_x0 + w_x0 / 2
    m = zigzag(16, scale=fit_scale(size / 2 - inset))
    ring, path, dot, bg = PALETTE[variant]
    mono = variant.startswith("mono")
    bands = BANDS.get(variant)
    rings = [0, 1, 2] if mono else [0, 1, 2, 3]
    # Draw iterates until one disappears under the minimizer dot (the last one drawn is a real
    # iterate inside the dot, so the final segment is a real step).
    n_pts = int(np.argmax(np.hypot(*m.iterates.T) < r_dot * 0.8)) + 1
    pts = m.iterates[:n_pts].copy()
    # The path starts at the rim of the hollow x0 ring, not at its center.
    d0 = pts[1] - pts[0]
    pts[0] = pts[0] + d0 / np.hypot(*d0) * (r_x0 + w_x0 / 2 - 0.2 * u)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" aria-label="{title}">',
        f"<title>{title}</title>",
    ]
    if bg:
        out.append(f'<rect width="{size}" height="{size}" rx="{size * 0.234:.2f}" fill="{bg}"/>')
    if bands:
        for i, fill in zip(rings, bands, strict=False):
            out.append(f'<ellipse {ellipse_attrs(m.levels[i], cx, cy)} fill="{fill}"/>')
    else:
        mid = f"cut-{variant}-{size}"
        gap = w_path + 2.6 * u
        x0 = m.iterates[0]
        out.append(
            f'<mask id="{mid}" maskUnits="userSpaceOnUse" x="0" y="0" width="{size}" height="{size}">'
            f'<rect width="{size}" height="{size}" fill="#fff"/>'
            f'<polyline points="{fmt_pts(pts, cx, cy)}" fill="none" stroke="#000" stroke-width="{gap:.3f}" '
            'stroke-linejoin="miter"/>'
            f'<circle cx="{cx + x0[0]:.3f}" cy="{cy - x0[1]:.3f}" r="{r_x0 + w_x0 + 1.2 * u:.3f}" fill="#000"/>'
            f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r_dot + 1.6 * u:.3f}" fill="#000"/></mask>'
        )
        els = "".join(
            f'<ellipse {ellipse_attrs(m.levels[i], cx, cy)} fill="none" stroke="{ring}" stroke-width="{w_ring * 0.85:.3f}"/>'
            for i in rings
        )
        out.append(f'<g mask="url(#{mid})">{els}</g>')
    out.append(
        f'<polyline points="{fmt_pts(pts, cx, cy)}" fill="none" stroke="{path}" '
        f'stroke-width="{w_path:.3f}" stroke-linecap="butt" stroke-linejoin="miter" stroke-miterlimit="4"/>'
    )
    x0 = m.iterates[0]
    out.append(
        f'<circle cx="{cx + x0[0]:.3f}" cy="{cy - x0[1]:.3f}" r="{r_x0:.3f}" fill="none" '
        f'stroke="{path}" stroke-width="{w_x0:.3f}"/>'
    )
    out.append(f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r_dot:.3f}" fill="{dot}"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# The 16-unit pixel master: the same theorem at s = 1.82, snapped to whole units. The exact
# iterates relative to x* are x0 = (-9.01, 6.43), x1 = (-4.60, 6.43), x2 = (-4.60, 3.28); on the
# grid they become (-9, 6), (-5, 6), (-5, 3): one horizontal and one vertical step (lengths 4
# and 3, ratio 0.75 against the exact r = 5/7). With a 2-unit stroke centered on whole units and
# a 4 x 4 dot centered on a whole unit, every edge lands on a pixel boundary at 16 px.
SMALL_C = (12, 10)  # x* in the 16-unit box (svg coordinates)
SMALL_PTS = [(-9, 6), (-5, 6), (-5, 3)]  # snapped x0, x1, x2 relative to x* (y up)


def small_points() -> list[tuple[int, int]]:
    exact = zigzag(2, scale=1.82).iterates
    snapped = np.array(SMALL_PTS, dtype=float)
    assert np.all(np.abs(exact - snapped) < 0.65), "snapped master drifted from the exact iterates"
    cx, cy = SMALL_C
    return [(cx + x, cy - y) for x, y in SMALL_PTS]


def small_mark_svg(variant: str = "color-light", px: int = 16, title: str = "numopt") -> str:
    """The 16-unit pixel master (use at 16-47 px). `px` only sets the default rendered size."""
    if variant not in PALETTE:
        raise ValueError(variant)
    _ring, path, dot, bg = PALETTE[variant]
    pts = small_points()
    cx, cy = SMALL_C
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="{px}" height="{px}" role="img" aria-label="{title}">',
        f"<title>{title}</title>",
    ]
    if bg:
        out.append(f'<rect width="16" height="16" rx="3.5" fill="{bg}"/>')
    pts_s = " ".join(f"{x},{y}" for x, y in pts)
    out.append(
        f'<polyline points="{pts_s}" fill="none" stroke="{path}" stroke-width="2" '
        'stroke-linecap="square" stroke-linejoin="miter"/>'
    )
    out.append(f'<rect x="{cx - 2}" y="{cy - 2}" width="4" height="4" rx="1.2" fill="{dot}"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ── Text to outlines ──────────────────────────────────────────────────────────────────


def text_path(
    text: str, font_file: Path, size: float, x: float, y: float, tracking: float = 0.0
) -> tuple[str, float]:
    """Return (SVG path d, advance width) for `text` set in `font_file` at `size` px, baseline y."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(str(font_file))
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upm = font["head"].unitsPerEm
    s = size / upm
    hmtx = font["hmtx"]
    kern = _kerning(font)
    pen = SVGPathPen(gs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    cursor = 0.0
    prev = None
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            continue
        if prev is not None:
            cursor += kern.get((prev, gname), 0)
        tp = TransformPen(pen, (s, 0, 0, -s, x + cursor * s, y))
        gs[gname].draw(tp)
        cursor += hmtx[gname][0] + tracking * upm
        prev = gname
    return pen.getCommands(), cursor * s


def _kerning(font) -> dict[tuple[str, str], int]:
    """Pair kerning from a simple GPOS PairPos (format 1 and 2) lookup; empty when absent."""
    pairs: dict[tuple[str, str], int] = {}
    if "GPOS" not in font:
        return pairs
    gpos = font["GPOS"].table
    for lookup in gpos.LookupList.Lookup:
        if lookup.LookupType not in (2, 9):
            continue
        for st in lookup.SubTable:
            if lookup.LookupType == 9:
                st = st.ExtSubTable
            if st.Format == 1:
                cov = st.Coverage.glyphs
                for i, ps in enumerate(st.PairSet):
                    for pv in ps.PairValueRecord:
                        v = getattr(pv.Value1, "XAdvance", 0) or 0
                        if v:
                            pairs.setdefault((cov[i], pv.SecondGlyph), v)
            elif st.Format == 2:
                cov = st.Coverage.glyphs
                c1 = st.ClassDef1.classDefs
                c2 = st.ClassDef2.classDefs
                second = {}
                for g, c in c2.items():
                    second.setdefault(c, []).append(g)
                for g1 in cov:
                    rec = st.Class1Record[c1.get(g1, 0)]
                    for c, r2 in enumerate(rec.Class2Record):
                        v = getattr(r2.Value1, "XAdvance", 0) or 0
                        if v:
                            for g2 in second.get(c, []):
                                pairs.setdefault((g1, g2), v)
    return pairs


def x_height(font_file: Path, size: float) -> float:
    from fontTools.ttLib import TTFont

    f = TTFont(str(font_file))
    return f["OS/2"].sxHeight / f["head"].unitsPerEm * size


def wordmark_svg(variant: str = "light", *, small: bool = False) -> str:
    """Mark + 'numopt' set in Newsreader Medium (outlined). The x-height is centered on x*.

    small: lockup for heights <= 40 px (uses the 16-unit pixel master of the mark).
    """
    ink = {"light": INK, "dark": PAPER_DARK, "mono-black": INK, "mono-white": "#ffffff"}[variant]
    mark_variant = {"light": "color-light", "dark": "color-dark"}.get(variant, variant)
    font = FONTS / "newsreader-latin-500-normal.woff"
    size = 74.0
    H = 112
    msize = 112  # the mark box; the ellipse is diagonal, so the box is larger than the text
    gap = 16
    if small:
        # The 16-unit master: x* sits at SMALL_C, the glyph's right edge at x = 14 units.
        k = msize / 16
        inner = small_mark_svg(mark_variant)
        inner = inner.split("\n", 2)[2].rsplit("</svg>", 1)[0]
        mark_g = f'<g transform="scale({k})">{inner}</g>'
        star_y = SMALL_C[1] * k
        text_x = 14 * k + gap
    else:
        inner = mark_svg(size=64, variant=mark_variant)
        inner = inner.split("\n", 2)[2].rsplit("</svg>", 1)[0]  # drop <svg> + <title>
        mark_g = f'<g transform="scale({msize / 64})">{inner}</g>'
        star_y = H / 2
        text_x = msize + gap
    d, adv = text_path("numopt", font, size=size, x=0, y=0, tracking=-0.012)
    baseline = star_y + x_height(font, size) / 2  # the x-height is centered on x*
    width = text_x + adv + 2
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.1f} {H}" width="{width:.0f}" height="{H}" role="img" aria-label="numopt">',
        "<title>numopt</title>",
        mark_g,
        f'<path transform="translate({text_x:.1f} {baseline:.2f})" d="{d}" fill="{ink}"/>',
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    LOGO.mkdir(parents=True, exist_ok=True)
    files = {
        "mark-color-light.svg": mark_svg(variant="color-light"),
        "mark-color-dark.svg": mark_svg(variant="color-dark"),
        "mark-mono-black.svg": mark_svg(variant="mono-black"),
        "mark-mono-white.svg": mark_svg(variant="mono-white"),
        "mark-tile.svg": mark_svg(size=512, variant="tile"),
        "mark-small-light.svg": small_mark_svg("color-light"),
        "mark-small-dark.svg": small_mark_svg("color-dark"),
        "mark-small-mono-black.svg": small_mark_svg("mono-black"),
        "mark-small-mono-white.svg": small_mark_svg("mono-white"),
        "favicon.svg": favicon_svg(),
        "wordmark-light.svg": wordmark_svg("light"),
        "wordmark-dark.svg": wordmark_svg("dark"),
        "wordmark-mono-black.svg": wordmark_svg("mono-black"),
        "wordmark-mono-white.svg": wordmark_svg("mono-white"),
        "wordmark-small-light.svg": wordmark_svg("light", small=True),
        "wordmark-small-dark.svg": wordmark_svg("dark", small=True),
    }
    for name, src in files.items():
        (LOGO / name).write_text(src)
    print(f"wrote {len(files)} files to {LOGO.relative_to(BRAND.parent.parent)}")
    # Raster icons for places that need PNG (apple-touch-icon, PWA manifest, package indexes).
    render = HERE / "render.mjs"
    for name, px in (("mark-tile.svg", 512), ("mark-tile.svg", 180)):
        out = LOGO / f"mark-tile-{px}.png"
        subprocess.run(
            ["node", str(render), str(LOGO / name), str(out), str(px), str(px), "--scale", "1"],
            check=True,
        )
    print("wrote mark-tile-512.png, mark-tile-180.png")


def favicon_svg() -> str:
    """The 16-unit pixel master on the ink tile: legible in light and dark browser tabs."""
    return small_mark_svg("tile", px=32)


if __name__ == "__main__":
    main()
