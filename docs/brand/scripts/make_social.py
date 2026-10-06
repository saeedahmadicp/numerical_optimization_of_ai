"""Social preview (GitHub "Social preview", Open Graph): 1280 x 640, SVG + PNG.

    .venv/bin/python docs/brand/scripts/make_social.py

The right half is the README hero's landscape and trajectories (same data, same colormap), bled
off the edge and faded into the ground; the left half is the lockup, the tagline and the counts.
The PNG is uploaded once to GitHub's settings and is not rebuilt on every release, so the method
count is rounded down to a multiple of ten ("150+", readme_facts.facts()["methods_floor"]): a
growing registry never makes the card wrong.
Upload docs/brand/logo/social-preview.png in the repository settings (Settings -> Social preview).
"""

from __future__ import annotations

import shutil
import subprocess

import make_hero as hero
import make_logo as logo
import numpy as np
import typeset as ts
from readme_facts import facts

W, H = 1280, 640
OUT = hero.HERE.parent / "logo"


def build(theme: str) -> str:
    th = hero.THEMES[theme]
    ground = "#0d0d0c" if theme == "dark" else "#f5f5f2"
    # Re-target the hero's plotting window to the right half of the card.
    # Equal scale on both axes: 704 / 652 px = 4.05 / 3.75 units.
    hero.LEFT = (636, -6, 704, 652)
    hero.DOMAIN = ((-2.05, 2.0), (-0.95, 2.80))
    p, runs = hero.DATA
    field = hero.landscape(p, theme, n_levels=17)
    order = hero.PAINT_ORDER
    dps = [hero.drawn_path(r) for r in sorted(runs, key=lambda r: order.index(r.cfg.id))]
    paths = hero.path_layers(dps, theme, animated=False, kmax=hero.K_MAX)

    fx = facts()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="numopt">',
        "<title>numopt — numerical optimization, iterate by iterate</title>",
        "<defs>"
        f'<linearGradient id="fade-{theme}" x1="636" x2="880" y1="0" y2="0" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff" stop-opacity="1"/>'
        "</linearGradient>"
        f'<mask id="m-{theme}"><rect x="0" y="0" width="{W}" height="{H}" fill="url(#fade-{theme})"/></mask>'
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{ground}"/>',
        f'<g mask="url(#m-{theme})">',
        *field,
        *paths,
        "</g>",
    ]
    # x0 and x* markers (no labels: the card is read at thumbnail size).
    for xy, kind in ((hero.X0, "x0"), ((1.0, 1.0), "xstar")):
        x, y = hero.to_px(np.array(xy))
        if kind == "x0":
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" stroke="{th["halo"]}" stroke-width="5"/>'
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" stroke="{th["text"]}" stroke-width="2"/>'
            )
        else:
            d = f"M{x - 8:.1f} {y:.1f}H{x + 8:.1f}M{x:.1f} {y - 8:.1f}V{y + 8:.1f}"
            parts.append(
                f'<path d="{d}" stroke="{th["halo"]}" stroke-width="6" stroke-linecap="round"/>'
            )
            parts.append(
                f'<path d="{d}" stroke="{th["text"]}" stroke-width="2" stroke-linecap="round"/>'
            )

    # Left column: lockup, tagline, facts, address.
    lx = 80
    mark = logo.mark_svg(size=64, variant="color-dark" if theme == "dark" else "color-light")
    mark_inner = mark.split("\n", 2)[2].rsplit("</svg>", 1)[0]
    parts.append(f'<g transform="translate({lx - 8} 66) scale(1.25)">{mark_inner}</g>')
    d, _ = ts.path_d([("numopt", "serif-medium", 46, 0.0)], lx + 82, 118, tracking=-0.012)
    parts.append(f'<path d="{d}" fill="{th["text"]}"/>')

    for i, line in enumerate(("Numerical optimization,", "iterate by iterate.")):
        d, _ = ts.path_d([(line, "serif", 62, 0.0)], lx, 290 + i * 72, tracking=-0.018)
        parts.append(f'<path d="{d}" fill="{th["text"] if i == 0 else th["text2"]}"/>')

    line = f"{fx['methods_floor']} methods · {fx['families']} families · every one cited and tested"
    d, _ = ts.path_d([(line, "inter", 21, 0.0)], lx, 420, tnum=True)
    parts.append(f'<path d="{d}" fill="{th["text2"]}"/>')
    d, _ = ts.path_d(
        [("Python reference · interactive labs in the browser", "inter", 21, 0.0)], lx, 452
    )
    parts.append(f'<path d="{d}" fill="{th["text2"]}"/>')

    # Legend chips for the drawn methods (identity never by color alone).
    x = lx
    for run in runs:
        color = hero.series(th, run.cfg)
        parts.append(f'<circle cx="{x + 5}" cy="{548}" r="5" fill="{color}"/>')
        d, w = ts.path_d(
            [(run.cfg.short, "inter-medium", 16, 0.0)],
            x + 16,
            554,
        )
        parts.append(f'<path d="{d}" fill="{th["text3"]}"/>')
        x += 16 + w + 22
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    hero.DATA = hero.run_methods()
    for theme in ("dark", "light"):
        svg = OUT / f"social-preview{'' if theme == 'dark' else '-light'}.svg"
        svg.write_text(build(theme))
        print("wrote", svg.relative_to(hero.REPO), f"{svg.stat().st_size / 1e3:.0f} kB")
        node = shutil.which("node")
        if node:
            png = svg.with_suffix(".png")
            subprocess.run(
                [node, str(hero.HERE / "render.mjs"), str(svg), str(png), "--scale", "1"],
                check=True,
            )
            print("wrote", png.relative_to(hero.REPO))


if __name__ == "__main__":
    main()
