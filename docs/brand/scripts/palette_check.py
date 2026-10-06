"""Compute every number brand.md quotes about color, and fail loudly if a gate is broken.

    .venv/bin/python docs/brand/scripts/palette_check.py            # prints Markdown tables
    .venv/bin/python docs/brand/scripts/palette_check.py --json     # machine-readable

Gates (same thresholds as the dataviz validator the web team used for --series-1..4):
  * text tokens >= 4.5:1 on bg, surface and sunken (WCAG AA body text)
  * accent >= 4.5:1 on every surface (it is used for links)
  * accent vs EVERY method color: normal-vision dE >= 15 and protan/deutan dE >= 8, so a focus
    ring or a link is never read as a fifth method (the same gates as the method pairs)
  * method colors, ALL pairs (paths cross): CVD dE >= 8 target (>= 6 floor with direct labels)
    under protan and deutan (Machado 2009, severity 1); normal-vision dE >= 15 (hard)
  * method colors >= 3:1 on the chart surface, or carry a direct label (relief rule)
  * method colors stay legible on every contour band: >= 2:1 vs the band behind them, with the
    2-px halo making up the rest (the halo is always drawn)
  * colormaps: OKLCH lightness strictly monotone
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations, pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colorlib import contrast, delta_e, oklch

THEMES = {
    "light": {
        "bg": "#f5f5f2",
        "surface": "#fcfcfb",
        "raised": "#ffffff",
        "sunken": "#eeeeea",
        "text": "#141413",
        "text-2": "#52514e",
        "text-3": "#6b6963",
        "accent (current)": "#3459d1",
        "accent: iris (proposed)": "#4b2ace",
        "good": "#0f7034",
        "warn": "#8f5500",
        "bad": "#c23434",
        "series": {
            "1 blue": "#2a78d6",
            "2 orange": "#eb6834",
            "3 aqua": "#1baf7a",
            "4 plum": "#882892",
        },
    },
    "dark": {
        "bg": "#0d0d0c",
        "surface": "#141413",
        "raised": "#1c1c1b",
        "sunken": "#0f0f0e",
        "text": "#f1f0eb",
        "text-2": "#bab9b0",
        "text-3": "#8a8982",
        "accent (current)": "#7f9cf5",
        "accent: iris (proposed)": "#b0a8fc",
        "good": "#3fbf6a",
        "warn": "#e3a336",
        "bad": "#ef6b6b",
        "series": {
            "1 blue": "#3987e5",
            "2 orange": "#d95926",
            "3 aqua": "#199e70",
            "4 plum": "#a13bab",
        },
    },
}

SURFACES = ("bg", "surface", "raised", "sunken")
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}


def contour_ends(theme: str) -> list[str]:
    import make_hero as hero

    return [hero.colormap(theme, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]


def report() -> dict:
    out: dict = {}
    failures: list[str] = []
    for name, th in THEMES.items():
        r: dict = {"text": {}, "accent": {}, "status": {}, "series": {}, "pairs": [], "field": {}}
        for tok in ("text", "text-2", "text-3"):
            r["text"][tok] = {s: round(contrast(th[tok], th[s]), 2) for s in SURFACES}
            if min(r["text"][tok].values()) < 4.5:
                failures.append(f"{name}: {tok} below 4.5:1")
        for tok in ("accent (current)", "accent: iris (proposed)"):
            r["accent"][tok] = {s: round(contrast(th[tok], th[s]), 2) for s in SURFACES}
            r["accent"][tok]["dE vs series-1"] = round(delta_e(th[tok], th["series"]["1 blue"]), 1)
            r["accent"][tok]["deutan dE vs series-1"] = round(
                delta_e(th[tok], th["series"]["1 blue"], "deutan"), 1
            )
        if min(v for k, v in r["accent"]["accent: iris (proposed)"].items() if k in SURFACES) < 4.5:
            failures.append(f"{name}: iris accent below 4.5:1")
        # Accent against every method color (it must never read as a fifth category).
        r["accent_vs_series"] = []
        for tok in ("accent (current)", "accent: iris (proposed)"):
            for label, hx in th["series"].items():
                row = {
                    "accent": tok,
                    "series": label,
                    "normal": round(delta_e(th[tok], hx), 1),
                    "protan": round(delta_e(th[tok], hx, "protan"), 1),
                    "deutan": round(delta_e(th[tok], hx, "deutan"), 1),
                    "tritan": round(delta_e(th[tok], hx, "tritan"), 1),
                }
                r["accent_vs_series"].append(row)
                if tok.startswith("accent: iris"):
                    if row["normal"] < 15:
                        failures.append(f"{name}: iris vs {label} normal dE {row['normal']} < 15")
                    if min(row["protan"], row["deutan"]) < 8:
                        failures.append(f"{name}: iris vs {label} CVD dE < 8")
        for tok in ("good", "warn", "bad"):
            r["status"][tok] = {s: round(contrast(th[tok], th[s]), 2) for s in ("bg", "surface")}
        field = contour_ends(name)
        for label, hx in th["series"].items():
            L, C, H = oklch(hx)
            r["series"][label] = {
                "hex": hx,
                "L": round(L, 3),
                "C": round(C, 3),
                "h": round(H, 1),
                "in band": BAND[name][0] <= L <= BAND[name][1],
                "vs surface": round(contrast(hx, th["surface"]), 2),
                "min vs contour band": round(min(contrast(hx, f) for f in field), 2),
            }
        hues = {k: v for k, v in th["series"].items()}
        worst = {"normal": math.inf, "cvd": math.inf, "tritan": math.inf}
        for (a, ha), (b, hb) in combinations(hues.items(), 2):
            row = {
                "pair": f"{a} / {b}",
                "normal": round(delta_e(ha, hb), 1),
                "protan": round(delta_e(ha, hb, "protan"), 1),
                "deutan": round(delta_e(ha, hb, "deutan"), 1),
                "tritan": round(delta_e(ha, hb, "tritan"), 1),
            }
            r["pairs"].append(row)
            worst["normal"] = min(worst["normal"], row["normal"])
            worst["cvd"] = min(worst["cvd"], row["protan"], row["deutan"])
            worst["tritan"] = min(worst["tritan"], row["tritan"])
        r["worst"] = worst
        if worst["normal"] < 15:
            failures.append(f"{name}: normal-vision dE {worst['normal']} < 15")
        if worst["cvd"] < 6:
            failures.append(f"{name}: CVD dE {worst['cvd']} < 6")
        r["field"]["contour L (t = 0 … 1)"] = [round(oklch(h)[0], 3) for h in field]
        Ls = r["field"]["contour L (t = 0 … 1)"]
        if not (all(a < b for a, b in pairwise(Ls)) or all(a > b for a, b in pairwise(Ls))):
            failures.append(f"{name}: contour map not monotone in L")
        out[name] = r
    out["failures"] = failures
    return out


def markdown(rep: dict) -> str:
    lines = []
    for name in ("light", "dark"):
        r = rep[name]
        lines.append(f"\n### {name.capitalize()} theme\n")
        lines.append("| Text token | bg | surface | raised | sunken |\n|---|---|---|---|---|")
        for tok, v in {**r["text"], **r["accent"]}.items():
            lines.append(f"| {tok} | " + " | ".join(f"{v[s]:.2f}" for s in SURFACES) + " |")
        lines.append(
            "\n| Method color | hex | OKLCH L / C / h | vs surface | min vs contour bands |\n|---|---|---|---|---|"
        )
        for label, v in r["series"].items():
            lines.append(
                f"| {label} | `{v['hex']}` | {v['L']:.3f} / {v['C']:.3f} / {v['h']:.0f} | {v['vs surface']:.2f} | {v['min vs contour band']:.2f} |"
            )
        lines.append(
            "\n| Pair (all pairs) | normal ΔE | protan ΔE | deutan ΔE | tritan ΔE |\n|---|---|---|---|---|"
        )
        for p in r["pairs"]:
            lines.append(
                f"| {p['pair']} | {p['normal']} | {p['protan']} | {p['deutan']} | {p['tritan']} |"
            )
        lines.append(
            "\n| Accent vs method color | normal ΔE (gate ≥ 15) | protan ΔE | deutan ΔE (gate ≥ 8) | tritan ΔE |\n|---|---|---|---|---|"
        )
        for a in r["accent_vs_series"]:
            lines.append(
                f"| {a['accent']} / {a['series']} | {a['normal']} | {a['protan']} | {a['deutan']} | {a['tritan']} |"
            )
        w = r["worst"]
        lines.append(
            f"\nWorst pair: normal ΔE {w['normal']}, protan/deutan ΔE {w['cvd']}, tritan ΔE {w['tritan']}."
        )
        lines.append(
            f"Contour map OKLCH L at t = 0, ¼, ½, ¾, 1: {r['field']['contour L (t = 0 … 1)']}."
        )
    lines.append("\nFailures: " + ("none" if not rep["failures"] else "; ".join(rep["failures"])))
    return "\n".join(lines)


if __name__ == "__main__":
    rep = report()
    print(json.dumps(rep, indent=1, ensure_ascii=False) if "--json" in sys.argv else markdown(rep))
    sys.exit(1 if rep["failures"] else 0)
