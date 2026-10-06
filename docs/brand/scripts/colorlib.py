"""Small, dependency-free color science used by the numopt brand scripts.

* sRGB <-> OKLab / OKLCH (Ottosson 2020).
* WCAG 2.x relative luminance and contrast ratio.
* Color-vision-deficiency simulation: Machado, Oliveira & Fernandes (2009), severity 1.0,
  applied in linear RGB (the same model and thresholds as the dataviz validator the web
  team used for `--series-1..4`, so numbers here and in web/src/ui/tokens.css agree).
* Delta E = Euclidean distance in OKLab x 100.
"""

from __future__ import annotations

import math

MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def hex_to_srgb(h: str) -> tuple[float, float, float]:
    h = h.strip().lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin2s(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def lin(h: str) -> tuple[float, float, float]:
    return tuple(s2lin(c) for c in hex_to_srgb(h))  # type: ignore[return-value]


def rel_lum(h: str) -> float:
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def blend(fg: str, bg: str, alpha: float) -> str:
    """Composite `fg` at `alpha` over opaque `bg` (in sRGB, as browsers do)."""
    f, b = hex_to_srgb(fg), hex_to_srgb(bg)
    return rgb_to_hex(tuple(alpha * x + (1 - alpha) * y for x, y in zip(f, b, strict=False)))  # type: ignore[arg-type]


def oklab_from_lin(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    l_ = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklab(h: str):
    return oklab_from_lin(lin(h))


def oklch(h: str) -> tuple[float, float, float]:
    L, a, b = oklab(h)
    return L, math.hypot(a, b), (math.degrees(math.atan2(b, a)) + 360) % 360


def oklch_to_hex(L: float, C: float, H: float) -> str | None:
    """OKLCH -> hex, or None when the color is outside the sRGB gamut."""
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    if min(r, g, bb) < -1e-4 or max(r, g, bb) > 1 + 1e-4:
        return None
    return rgb_to_hex(tuple(lin2s(c) for c in (r, g, bb)))  # type: ignore[arg-type]


def rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{round(min(1, max(0, c)) * 255):02x}" for c in rgb)


def simulate(h: str, kind: str):
    M = MACHADO[kind]
    r, g, b = lin(h)
    return tuple(min(1.0, max(0.0, row[0] * r + row[1] * g + row[2] * b)) for row in M)


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)
