"""README hero: four optimizers on the Rosenbrock valley, computed with numopt.

    .venv/bin/python docs/brand/scripts/make_hero.py            # SVGs (+ raster fallbacks in build/)
    .venv/bin/python docs/brand/scripts/make_hero.py --no-raster

Every trajectory is a real `numopt.run(...)` trace from the textbook start x0 = (-1.2, 1)
(Rosenbrock 1960; Nocedal & Wright 2006, Sec. 2.2). All four runs use ONE stopping test, the
first k with ||grad f(x_k)||_2 <= 1e-8, inside one 20,000-iteration budget; every other parameter
is the registered default. (BFGS and Newton test the inf-norm internally, so they run with a
tighter internal tolerance and their trace is cut at the first iterate that passes the common
test. The methods are deterministic, so the cut trace is exactly the run with that test.)

The left panel is the level-set landscape with the iterates, at equal scale on both axes; the
right panel is f(x_k) - f* against k on log-log axes, so linear, superlinear and quadratic
convergence read as different shapes. Diamonds mark x_k at k = 10, 100, 1000, 10000 on both
panels, so each method's progress along the valley lines up with the log-k axis.

Outputs (docs/brand/readme-hero/), each in light and dark:
    hero-{theme}.svg                    wide static figure (960 x 640), docs and slides
    hero-animated-{theme}.svg           wide, SMIL on one clock linear in log k; plays once, then
                                        holds the final frame; prefers-reduced-motion shows it
    hero-stacked-{theme}.svg            narrow static figure (540 wide), panels stacked
    hero-stacked-animated-{theme}.svg   narrow animated figure (README on phones)
Raster fallbacks (PNG, play-once WebP) go to docs/brand/build/ (git-ignored).
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import contourpy
import numpy as np
import typeset as ts
from colorlib import blend, rgb_to_hex

import numopt
from numopt import problems

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "readme-hero"
BUILD = HERE.parent / "build" / "readme-hero"
REPO = HERE.parents[2]

# ── What is drawn ─────────────────────────────────────────────────────────────────────
PROBLEM = "rosenbrock"
X0 = (-1.2, 1.0)
DOMAIN = ((-2.0, 2.0), (-1.0, 2.75))  # 4 x 3.75 units; panels keep this aspect (equal scale)
GTOL = 1e-8  # the common stopping test: ||grad f(x_k)||_2 <= GTOL
BUDGET = 20_000  # the common iteration budget
MILESTONES = (10, 100, 1_000, 10_000)


@dataclass
class MethodCfg:
    id: str
    label: str  # legend: textbook name, variant in parentheses (brand.md, "Voice")
    short: str  # social card chips
    slot: int  # 1..4 = --series-1..4
    params: dict = field(default_factory=dict)
    dots: bool = False  # mark every iterate (short traces only)


METHODS = [
    MethodCfg("gradient_descent", "Gradient descent (Armijo backtracking)", "Gradient descent", 1),
    MethodCfg("momentum", "Heavy-ball momentum", "Momentum", 2),
    MethodCfg("bfgs", "BFGS", "BFGS", 3, dots=True),
    MethodCfg("pure_newton", "Newton", "Newton", 4, dots=True),
]

# Paint order in the landscape: the long, smooth crawlers share the valley floor, so the slowest
# is painted last with the thinnest stroke; the iterate chains (BFGS, Newton) sit underneath with
# dots that stay visible around it.
PAINT_ORDER = ["momentum", "bfgs", "pure_newton", "gradient_descent"]
STROKE = {"momentum": 2.2, "bfgs": 2.0, "pure_newton": 2.0, "gradient_descent": 1.5}

# ── Themes (mirror web/src/ui/tokens.css; brand.md documents every value) ───────────
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        border=blend("#141413", "#fcfcfb", 0.10),
        text="#141413",
        text2="#52514e",
        text3="#6b6963",
        grid=blend("#141413", "#fcfcfb", 0.07),
        axis=blend("#141413", "#fcfcfb", 0.22),
        halo="#fcfcfb",
        iso="rgba(30,40,60,0.15)",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#882892"],
        playhead=blend("#141413", "#fcfcfb", 0.45),
    ),
    "dark": dict(
        surface="#141413",
        border=blend("#fffffa", "#141413", 0.10),
        text="#f1f0eb",
        text2="#bab9b0",
        text3="#8a8982",
        grid=blend("#fffffa", "#141413", 0.06),
        axis=blend("#fffffa", "#141413", 0.2),
        halo="#141413",
        iso="rgba(220,230,255,0.10)",
        series=["#3987e5", "#d95926", "#199e70", "#a13bab"],
        playhead=blend("#fffffa", "#141413", 0.5),
    ),
}


def series(th: dict, cfg: MethodCfg) -> str:
    return th["series"][cfg.slot - 1]


# Contour colormaps: the same OKLCH anchors as web/src/ui/colors.ts (CONTOUR_MAPS), so the hero
# and the lab paint one landscape with one color. t = 0 is the lowest f.
CONTOUR_ANCHORS = {
    "light": [
        (0, 0.8, 0.032, 212),
        (0.3, 0.868, 0.034, 196),
        (0.62, 0.93, 0.03, 150),
        (1, 0.982, 0.03, 88),
    ],
    "dark": [
        (0, 0.42, 0.034, 212),
        (0.32, 0.33, 0.03, 200),
        (0.66, 0.255, 0.022, 170),
        (1, 0.195, 0.012, 95),
    ],
}

# ── Layouts ───────────────────────────────────────────────────────────────────────────
# The README column is 830-1012 px wide on desktop. The wide figure is laid out at 960 units,
# so 1 unit is 0.86-1.05 px there and the smallest type (14 units) is 12-15 px. On a phone
# (<= 640 px viewport) the README shows the stacked figure, laid out at 540 units with 17-unit
# minimum type: about 11 px at a 360 px column.


@dataclass
class Layout:
    name: str
    W: int
    H: int
    left: tuple[float, float, float, float]  # landscape x, y, w, h (h = w * 3.75 / 4)
    right: tuple[float, float, float, float]  # convergence plot area
    head_left: tuple[float, float]  # baseline of the landscape title
    head_right: tuple[float, float]  # baseline of the convergence title
    formula_below: bool  # set the Rosenbrock formula on its own line
    legend_y: float
    legend_rows: bool  # one method per row (stacked) instead of one row
    fs: dict  # type sizes in layout units


WIDE = Layout(
    name="wide",
    W=960,
    H=650,
    left=(58, 76, 500, 468.75),
    right=(644, 76, 290, 468.75),
    head_left=(58, 48),
    head_right=(644, 48),
    formula_below=False,
    legend_y=630,
    legend_rows=False,
    fs=dict(title=16, formula=19, tick=14, axis=15, legend=14.5, note=14, label=17, rate=16),
)

STACKED = Layout(
    name="stacked",
    W=540,
    H=1206,
    left=(54, 96, 462, 433.125),
    right=(96, 676, 420, 300),
    head_left=(24, 38),
    head_right=(24, 642),
    formula_below=True,
    legend_y=1100,
    legend_rows=True,
    fs=dict(title=19, formula=21, tick=17, axis=18, legend=18, note=17, label=20, rate=18),
)

# Module state set by use_layout(); make_social.py also overrides LEFT and DOMAIN.
LAYOUT = WIDE
W, H = WIDE.W, WIDE.H
LEFT = WIDE.left
RIGHT = WIDE.right
FS = WIDE.fs


def use_layout(lay: Layout) -> None:
    global LAYOUT, W, H, LEFT, RIGHT, FS
    LAYOUT, W, H, LEFT, RIGHT, FS = lay, lay.W, lay.H, lay.left, lay.right, lay.fs


# Animation clock: k(tau) is linear in log k, so the convergence playhead moves at constant speed.
# The figure plays once and then holds its final frame (brand.md, "Motion": no endless loops).
T_PRE, T_DRAW = 0.5, 9.0
T_END = T_PRE + T_DRAW
N_KEYS = 150
K_MAX = 20_000  # right end of the log-k axis (>= BUDGET)
Y_DEC = (-24, 4)  # log10 range of f - f* (Newton's second iterate has f = 1.4e3)


# ── Data ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Run:
    cfg: MethodCfg
    xs: np.ndarray  # (n+1, 2) iterates
    fs: np.ndarray  # (n+1,) f(x_k)
    converged: bool  # the common test passed within the budget
    n_iter: int


def run_methods() -> tuple[object, list[Run]]:
    p = problems.get(PROBLEM)
    runs = []
    for cfg in METHODS:
        spec = numopt.get_method(cfg.id)
        bounds = {ps.name: ps for ps in spec.params}
        # Internal tolerance at its floor, so the method never stops before the common test.
        params = {"gtol": bounds["gtol"].min, "max_iter": min(BUDGET, int(bounds["max_iter"].max))}
        params.update(cfg.params)
        r = numopt.run(cfg.id, p, x0=list(X0), **params)
        xs = np.array([s.x for s in r.trace], dtype=float)
        fs = np.array([s.fun for s in r.trace], dtype=float)
        g2 = np.array([np.linalg.norm(p.grad(x)) for x in xs])
        hit = np.flatnonzero(g2 <= GTOL)
        if len(hit):
            k = int(hit[0])
            runs.append(Run(cfg, xs[: k + 1], fs[: k + 1], True, k))
        else:
            runs.append(Run(cfg, xs, fs, False, len(xs) - 1))
    return p, runs


# ── Color ─────────────────────────────────────────────────────────────────────────────


def _oklab_to_hex(L: float, a: float, b: float) -> str:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def gamma(c: float) -> float:
        c = min(1.0, max(0.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return rgb_to_hex((gamma(r), gamma(g), gamma(bb)))


def colormap(theme: str, t: float) -> str:
    """Port of interpAnchors() in web/src/ui/colors.ts (interpolates in OKLab a, b)."""
    anchors = CONTOUR_ANCHORS[theme]
    x = min(1.0, max(0.0, t))
    i = 0
    while i < len(anchors) - 2 and x > anchors[i + 1][0]:
        i += 1
    t0, L0, C0, h0 = anchors[i]
    t1, L1, C1, h1 = anchors[i + 1]
    u = 0.0 if t1 == t0 else (x - t0) / (t1 - t0)
    a0, b0 = C0 * math.cos(math.radians(h0)), C0 * math.sin(math.radians(h0))
    a1, b1 = C1 * math.cos(math.radians(h1)), C1 * math.sin(math.radians(h1))
    return _oklab_to_hex(L0 + (L1 - L0) * u, a0 + (a1 - a0) * u, b0 + (b1 - b0) * u)


# ── Typography helpers (brand.md, "Numerals and notation") ────────────────────────────


def num(v: float, digits: int = 2) -> str:
    """Fixed-point with U+2212 for the minus sign."""
    return f"{v:.{digits}f}".replace("-", "−")


def count(n: int) -> str:
    return f"{n:,}"


def tex_num(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}"


# ── Geometry helpers ──────────────────────────────────────────────────────────────────


def to_px(xy: np.ndarray) -> np.ndarray:
    (x0, x1), (y0, y1) = DOMAIN
    lx, ly, lw, lh = LEFT
    px = lx + (xy[..., 0] - x0) / (x1 - x0) * lw
    py = ly + lh - (xy[..., 1] - y0) / (y1 - y0) * lh
    return np.stack([px, py], axis=-1)


def rdp(P: np.ndarray, eps: float) -> np.ndarray:
    """Indices kept by Ramer–Douglas–Peucker (iterative)."""
    n = len(P)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = float(np.hypot(*ab))
        seg = P[i + 1 : j]
        if L < 1e-12:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def d_poly(P: np.ndarray, closed: bool = False) -> str:
    s = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in P)
    return s + ("Z" if closed else "")


def fnum(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def inside(P: np.ndarray, pad: float = 0.0) -> np.ndarray:
    lx, ly, lw, lh = LEFT
    return (
        (P[:, 0] >= lx - pad)
        & (P[:, 0] <= lx + lw + pad)
        & (P[:, 1] >= ly - pad)
        & (P[:, 1] <= ly + lh + pad)
    )


def frame_crossing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Point where the segment a -> b crosses the panel frame (a inside, b outside, or vice versa)."""
    _lx, _ly, _lw, _lh = LEFT
    lo, hi = 0.0, 1.0
    ina = bool(inside(a[None])[0])
    for _ in range(60):
        mid = (lo + hi) / 2
        p = a + (b - a) * mid
        if bool(inside(p[None])[0]) == ina:
            lo = mid
        else:
            hi = mid
    return a + (b - a) * lo


# ── Animation helpers ─────────────────────────────────────────────────────────────────


def clock_time(k: float, kmax: float = K_MAX) -> float:
    if k <= 1:
        return T_PRE * k
    return T_PRE + T_DRAW * math.log(k) / math.log(kmax)


def k_of_tau(tau: np.ndarray, kmax: float = K_MAX) -> np.ndarray:
    """The shared clock: k rises 0 -> 1 during T_PRE, then log k is linear up to log kmax."""
    return np.where(tau < T_PRE, tau / T_PRE, np.power(kmax, np.clip((tau - T_PRE) / T_DRAW, 0, 1)))


def appear(t: float) -> str:
    """Opacity 0 -> 1 at time t, then held (plays once)."""
    return f'<set attributeName="opacity" to="1" begin="{max(t, 0.001):.3f}s" fill="freeze"/>'


def vanish(t: float) -> str:
    return f'<set attributeName="opacity" to="0" begin="{t:.3f}s" fill="freeze"/>'


# ── Landscape ─────────────────────────────────────────────────────────────────────────


def landscape(p, theme: str, n_levels: int = 15) -> list[str]:
    """Filled sublevel sets {f <= c_i}, painted from the highest level down, each with its iso-line.

    Levels are uniform in t = log(f - f_min + delta) over the plotted domain (the web's 'log'
    level scale, contourField.ts), so the steep walls and the flat valley both get bands.
    """
    (x0, x1), (y0, y1) = DOMAIN
    nx, ny = 480, 450
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: float(p.f(np.array([a, b]))))(X, Y)
    fmin, fmax = 0.0, float(Z.max())
    delta = (fmax - fmin) * 2e-4
    T = np.log(Z - fmin + delta)
    t_lo, t_hi = math.log(delta), math.log(fmax - fmin + delta)
    levels = [t_lo + (t_hi - t_lo) * (i / n_levels) for i in range(1, n_levels)]
    gen = contourpy.contour_generator(xs, ys, T, fill_type=contourpy.FillType.OuterCode)
    out = []
    lx, ly, lw, lh = LEFT
    out.append(
        f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="{colormap(theme, 1.0)}"/>'
    )
    for i in reversed(range(len(levels))):
        polys, _codes = gen.filled(-1e9, levels[i])
        color = colormap(theme, (i + 0.5) / n_levels)
        parts = []
        for poly in polys:
            P = to_px(poly)
            keep = rdp(P, 0.35)
            if len(keep) < 3:
                continue
            parts.append(d_poly(P[keep], closed=True))
        if not parts:
            continue
        out.append(
            f'<path d="{" ".join(parts)}" fill="{color}" fill-rule="evenodd" '
            f'stroke="{THEMES[theme]["iso"]}" stroke-width="0.8"/>'
        )
    return out


# ── Trajectories ──────────────────────────────────────────────────────────────────────


@dataclass
class DrawnPath:
    run: Run
    d: str
    kept: np.ndarray
    cum: np.ndarray  # arc length (px) at every iterate, along the drawn polyline
    length: float


def drawn_path(run: Run) -> DrawnPath:
    P = to_px(run.xs)
    keep = rdp(P, 0.3) if len(P) > 60 else np.arange(len(P))
    Q = P[keep]
    seg = np.hypot(*np.diff(Q, axis=0).T)
    cum_kept = np.concatenate([[0.0], np.cumsum(seg)])
    cum = np.interp(np.arange(len(P)), keep, cum_kept)
    return DrawnPath(run, d_poly(Q), keep, cum, float(cum_kept[-1]))


def diamond(x: float, y: float, r: float, fill: str, halo: str, extra: str = "") -> str:
    d = f"M{x:.1f} {y - r:.1f}L{x + r:.1f} {y:.1f}L{x:.1f} {y + r:.1f}L{x - r:.1f} {y:.1f}Z"
    return f'<path d="{d}" fill="{fill}" stroke="{halo}" stroke-width="1.4"{extra}/>'


def chevron(p: np.ndarray, direction: np.ndarray, color: str, halo: str, size: float = 6.0) -> str:
    """An open arrowhead at p pointing along `direction`."""
    u = direction / max(float(np.hypot(*direction)), 1e-9)
    n = np.array([-u[1], u[0]])
    tip = p + u * size * 0.55
    a = tip - u * size + n * size * 0.75
    b = tip - u * size - n * size * 0.75
    d = f"M{a[0]:.1f} {a[1]:.1f}L{tip[0]:.1f} {tip[1]:.1f}L{b[0]:.1f} {b[1]:.1f}"
    return (
        f'<path d="{d}" fill="none" stroke="{halo}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def path_layers(
    dp_list: list[DrawnPath], theme: str, *, animated: bool, kmax: float = K_MAX
) -> list[str]:
    th = THEMES[theme]
    out: list[str] = []
    taus = np.linspace(0, T_END, N_KEYS)
    kt = ";".join(fnum(v) for v in np.round(taus / T_END, 4))
    ks = k_of_tau(taus, kmax)
    for dp in dp_list:
        run = dp.run
        color = series(th, run.cfg)
        width = STROKE.get(run.cfg.id, 1.8)
        P = to_px(run.xs)
        ok = inside(P)
        if run.cfg.dots:
            # Iterate chains: one segment per step, each appearing when the clock reaches k + 1.
            # Segments with an end outside the view are dashed in every frame.
            for i in range(len(P) - 1):
                a, b = P[i], P[i + 1]
                dash = not (ok[i] and ok[i + 1])
                seg = f"M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}"
                body = (
                    f'<path d="{seg}" fill="none" stroke="{th["halo"]}" stroke-opacity="0.85" stroke-width="{width + 3:.1f}" stroke-linecap="round"/>'
                    f'<path d="{seg}" fill="none" stroke="{color}" stroke-width="{width - (0.3 if dash else 0):.1f}" stroke-linecap="round"'
                    + (' stroke-dasharray="5 5"' if dash else "")
                    + "/>"
                )
                if animated:
                    out.append(f'<g opacity="0">{appear(clock_time(i + 1, kmax))}{body}</g>')
                else:
                    out.append(body)
            for k in range(1, len(P)):
                if not ok[k]:
                    continue
                x, y = P[k]
                c = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{color}" stroke="{th["halo"]}" stroke-width="1.3"/>'
                out.append(
                    f'<g opacity="0">{appear(clock_time(k, kmax))}{c}</g>' if animated else c
                )
        else:
            dash_attr, anim = "", ""
            if animated:
                frac = np.interp(np.minimum(ks, run.n_iter), np.arange(len(dp.cum)), dp.cum) / max(
                    dp.length, 1e-9
                )
                vals = ";".join(fnum(1 - f) for f in np.round(np.clip(frac, 0, 1), 4))
                dash_attr = ' pathLength="1" stroke-dasharray="1 1" stroke-dashoffset="1"'
                anim = (
                    f'<animate attributeName="stroke-dashoffset" dur="{T_END}s" fill="freeze" '
                    f'calcMode="linear" keyTimes="{kt}" values="{vals}"/>'
                )
            out.append(
                f'<path d="{dp.d}" fill="none" stroke="{th["halo"]}" stroke-opacity="0.8" stroke-width="{width + 3:.1f}" '
                f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}>{anim}</path>'
            )
            out.append(
                f'<path d="{dp.d}" fill="none" stroke="{color}" stroke-width="{width}" '
                f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}>{anim}</path>'
            )
            if animated:
                # The moving head x_k on the shared clock; it leaves when the clock stops, so the
                # held final frame equals the static figure.
                kp = ";".join(fnum(v) for v in np.round(np.clip(frac, 0, 1), 4))
                out.append(
                    f'<circle r="4.4" fill="{color}" stroke="{th["halo"]}" stroke-width="1.6">'
                    f'<animateMotion dur="{T_END}s" fill="freeze" calcMode="linear" '
                    f'keyTimes="{kt}" keyPoints="{kp}" path="{dp.d}"/>{vanish(T_END)}</circle>'
                )
    # Milestones on top of every path: x_k at k = 10, 100, 1000, 10000.
    for dp in dp_list:
        run = dp.run
        color = series(th, run.cfg)
        for k in MILESTONES:
            if k >= run.n_iter:
                continue
            x, y = to_px(run.xs[k])
            if not inside(np.array([[x, y]]))[0]:
                continue
            m = diamond(x, y, 4.6, color, th["halo"])
            out.append(f'<g opacity="0">{appear(clock_time(k, kmax))}{m}</g>' if animated else m)
    return out


# ── Convergence chart ─────────────────────────────────────────────────────────────────


def conv_xy(k: np.ndarray, f: np.ndarray) -> np.ndarray:
    rx, ry, rw, rh = RIGHT
    x = rx + np.log10(k) / math.log10(K_MAX) * rw
    lf = np.log10(np.maximum(f, 10.0 ** Y_DEC[0]))
    y = ry + rh - (lf - Y_DEC[0]) / (Y_DEC[1] - Y_DEC[0]) * rh
    return np.stack([x, y], axis=-1)


def convergence(runs: list[Run], theme: str, *, animated: bool) -> list[str]:
    th = THEMES[theme]
    rx, ry, rw, rh = RIGHT
    fs = FS
    out = []
    for e in range(Y_DEC[0], Y_DEC[1] + 1, 8):
        y = ry + rh - (e - Y_DEC[0]) / (Y_DEC[1] - Y_DEC[0]) * rh
        out.append(f'<path d="M{rx} {y:.1f}H{rx + rw}" stroke="{th["grid"]}" stroke-width="1"/>')
        d, _ = ts.path_d(
            ts.math(f"10^{{{e}}}" if e else "1", fs["tick"]),
            rx - 9,
            y + fs["tick"] * 0.36,
            anchor="end",
        )
        out.append(f'<path d="{d}" fill="{th["text3"]}"/>')
    for e in range(0, 5):
        x = rx + e / math.log10(K_MAX) * rw
        out.append(f'<path d="M{x:.1f} {ry}V{ry + rh}" stroke="{th["grid"]}" stroke-width="1"/>')
        label = "1" if e == 0 else ("10" if e == 1 else f"10^{e}")
        d, _ = ts.path_d(ts.math(label, fs["tick"]), x, ry + rh + fs["tick"] + 9, anchor="middle")
        out.append(f'<path d="{d}" fill="{th["text3"]}"/>')
    out.append(
        f'<path d="M{rx} {ry + rh}H{rx + rw}M{rx} {ry}V{ry + rh}" stroke="{th["axis"]}" stroke-width="1"/>'
    )
    d, _ = ts.path_d(
        ts.text("iteration ", fs["axis"] - 1, "inter") + ts.math("k \\ge 1", fs["axis"] + 1),
        rx + rw,
        ry + rh + fs["tick"] + 9 + fs["axis"] + 13,
        anchor="end",
    )
    out.append(f'<path d="{d}" fill="{th["text3"]}"/>')

    body = []
    labels: list[tuple[float, str]] = []  # rate annotations, shown when their run ends
    for run in runs:
        color = series(th, run.cfg)
        k = np.arange(1, len(run.fs))
        P = conv_xy(k.astype(float), run.fs[1:])
        keep = rdp(P, 0.25)
        width = 2.0
        body.append(
            f'<path d="{d_poly(P[keep])}" fill="none" stroke="{th["halo"]}" stroke-opacity="0.85" stroke-width="{width + 2.8:.1f}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        body.append(
            f'<path d="{d_poly(P[keep])}" fill="none" stroke="{color}" stroke-width="{width}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for km in MILESTONES:
            if km < run.n_iter:
                mx, my = P[km - 1]
                body.append(diamond(mx, my, 4.2, color, th["halo"]))
        ex, ey = P[-1]
        note = {"pure_newton": "quadratic", "bfgs": "superlinear"}.get(run.cfg.id)
        if note:
            # Newton ends left of BFGS's vertical drop: label it on its left so the two never touch.
            left = run.cfg.id == "pure_newton"
            labels.append(
                (
                    clock_time(run.n_iter),
                    fill_path(
                        [(note, "serif-italic", fs["rate"], 0.0)],
                        ex - 9 if left else ex + 9,
                        ey + 5,
                        th["text2"],
                        "end" if left else "start",
                        th["surface"],
                    ),
                )
            )
        if run.converged:
            body.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.8" fill="{color}" stroke="{th["halo"]}" stroke-width="1.5"/>'
            )
        else:
            body.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.6" fill="{th["surface"]}" stroke="{color}" stroke-width="1.8"/>'
            )
    if animated:
        # One clip rectangle sweeps the log-k axis at constant speed: that is the shared clock.
        clip = f"conv-clip-{theme}-{LAYOUT.name}"
        out.append(
            f'<clipPath id="{clip}"><rect x="{rx - 90}" y="{ry - 14}" height="{rh + 28}" width="0">'
            f'<animate attributeName="width" dur="{T_END}s" fill="freeze" calcMode="linear" '
            f'keyTimes="0;{T_PRE / T_END:.4f};1" values="0;96;{rw + 210}"/>'
            f"</rect></clipPath>"
        )
        out.append(f'<g clip-path="url(#{clip})">' + "".join(body) + "</g>")
        out += [f'<g opacity="0">{appear(t)}{lab}</g>' for t, lab in labels]
        out.append(
            f'<path d="M{rx} {ry - 6}V{ry + rh}" stroke="{th["playhead"]}" stroke-width="1" stroke-dasharray="3 3">'
            f'<animateTransform attributeName="transform" type="translate" dur="{T_END}s" fill="freeze" '
            f'calcMode="linear" keyTimes="0;{T_PRE / T_END:.4f};1" values="0 0;0 0;{rw} 0"/>{vanish(T_END)}</path>'
        )
    else:
        out += body + [lab for _t, lab in labels]
    return out


# ── Annotations, header, legend ───────────────────────────────────────────────────────


def fill_path(runs, x, y, color, anchor="start", halo=None, tnum=False) -> str:
    d, _ = ts.path_d(runs, x, y, anchor=anchor, tnum=tnum)
    if halo:
        return (
            f'<path d="{d}" fill="{color}" stroke="{halo}" stroke-width="3.2" stroke-linejoin="round" '
            'paint-order="stroke"/>'
        )
    return f'<path d="{d}" fill="{color}"/>'


def axes_left(theme: str) -> list[str]:
    """Ticks and axis names of the landscape, outside the frame; equal scale on x and y."""
    th = THEMES[theme]
    lx, ly, lw, lh = LEFT
    fs = FS
    out = []
    ty = ly + lh + fs["tick"] + 9
    for xv in (-1, 0, 1):
        x, _ = to_px(np.array([xv, 0.0]))
        out.append(f'<path d="M{x:.1f} {ly + lh}v5" stroke="{th["axis"]}" stroke-width="1"/>')
        out.append(fill_path(ts.math(str(xv), fs["tick"]), x, ty, th["text3"], "middle"))
    for yv in (0, 1, 2):
        _, y = to_px(np.array([0.0, yv]))
        out.append(f'<path d="M{lx} {y:.1f}h-5" stroke="{th["axis"]}" stroke-width="1"/>')
        out.append(
            fill_path(
                ts.math(str(yv), fs["tick"]), lx - 9, y + fs["tick"] * 0.36, th["text3"], "end"
            )
        )
    out.append(fill_path(ts.math("x", fs["axis"] + 2), lx + lw, ty, th["text2"], "end"))
    out.append(
        fill_path(ts.math("y", fs["axis"] + 2), lx - 9, ly + fs["axis"] + 2, th["text2"], "end")
    )
    # Milestone key, under the landscape.
    kx, ky = lx, ty + fs["axis"] + 13
    out.append(diamond(kx + 5, ky - fs["note"] * 0.36, 4.6, th["text2"], th["surface"]))
    key = (
        ts.math("\\mathbf{x}_k", fs["note"] + 2)
        + ts.text(" at ", fs["note"], "inter")
        + ts.math("k = 10, 10^2, 10^3, 10^4", fs["note"] + 2)
        + ts.text("  ·  equal scale on both axes", fs["note"], "inter")
    )
    out.append(fill_path(key, kx + 16, ky, th["text3"]))
    return out


def annotations(
    runs: list[Run], theme: str, *, animated: bool = False, kmax: float = K_MAX
) -> list[str]:
    th = THEMES[theme]
    _lx, ly, _lw, lh = LEFT
    fs = FS
    out = []
    # x0 (hollow ring) and x* (cross with coordinates).
    sx, sy = to_px(np.array(X0))
    out.append(
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6.5" fill="none" stroke="{th["halo"]}" stroke-width="4.5"/>'
    )
    out.append(
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6.5" fill="none" stroke="{th["text"]}" stroke-width="1.8"/>'
    )
    out.append(
        fill_path(
            ts.math("\\mathbf{x}_0", fs["label"] + 1),
            sx - 13,
            sy - 11,
            th["text"],
            "end",
            th["halo"],
        )
    )
    mx, my = to_px(np.array([1.0, 1.0]))
    out.append(
        f'<path d="M{mx - 7:.1f} {my:.1f}H{mx + 7:.1f}M{mx:.1f} {my - 7:.1f}V{my + 7:.1f}" stroke="{th["halo"]}" stroke-width="5" stroke-linecap="round"/>'
        f'<path d="M{mx - 7:.1f} {my:.1f}H{mx + 7:.1f}M{mx:.1f} {my - 7:.1f}V{my + 7:.1f}" stroke="{th["text"]}" stroke-width="1.8" stroke-linecap="round"/>'
    )
    out.append(
        fill_path(
            ts.math("\\mathbf{x}^\\star = (1, 1)", fs["label"]),
            mx + 13,
            my + fs["label"] + 6,
            th["text"],
            "start",
            th["halo"],
        )
    )
    # Steps that leave the view: an outward chevron where the step leaves, an inward chevron where
    # the next step comes back, and the off-view iterate labeled at the exit point.
    for run in runs:
        P = to_px(run.xs)
        ok = inside(P)
        color = series(th, run.cfg)
        for i in range(1, len(P) - 1):
            if ok[i] or not (ok[i - 1] and ok[i + 1]):
                continue
            a, b, c = P[i - 1], P[i], P[i + 1]
            exit_p = frame_crossing(a, b)
            entry_p = frame_crossing(c, b)
            xi = run.xs[i]
            side = "below" if b[1] > ly + lh else "above" if b[1] < ly else "outside"
            # Right edge of the two-line label: clear of the dashed step at the label's top line.
            top = ly + lh - 14 - fs["note"] * 2.6
            x_top = (
                a[0] + (top - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) if b[1] != a[1] else exit_p[0]
            )
            lab_x = min(exit_p[0], x_top) - 12
            note = [
                chevron(exit_p - (b - a) / np.hypot(*(b - a)) * 7, b - a, color, th["halo"]),
                chevron(entry_p + (c - b) / np.hypot(*(c - b)) * 9, c - b, color, th["halo"]),
                fill_path(
                    ts.math(
                        f"\\mathbf{{x}}_{i} = ({tex_num(xi[0])}, {tex_num(xi[1])})", fs["note"] + 2
                    ),
                    lab_x,
                    ly + lh - 14 - fs["note"] * 1.35,
                    th["text"],
                    "end",
                    th["halo"],
                ),
                fill_path(
                    ts.text(f"{run.cfg.short}, {side} the view", fs["note"], "inter"),
                    lab_x,
                    ly + lh - 14,
                    th["text2"],
                    "end",
                    th["halo"],
                ),
            ]
            if animated:
                out.append(f'<g opacity="0">{appear(clock_time(i, kmax))}' + "".join(note) + "</g>")
            else:
                out += note
    return out


def header(theme: str) -> list[str]:
    th = THEMES[theme]
    fs = FS
    out = []
    hx, hy = LAYOUT.head_left
    d, w = ts.path_d(ts.text("Rosenbrock", fs["title"], "inter-semibold"), hx, hy)
    out.append(f'<path d="{d}" fill="{th["text"]}"/>')
    formula = ts.math("f(x, y) = (1 - x)^2 + 100\\,(y - x^2)^2", fs["formula"])
    if LAYOUT.formula_below:
        out.append(fill_path(formula, hx, hy + fs["formula"] + 12, th["text2"]))
    else:
        out.append(fill_path(formula, hx + w + 14, hy, th["text2"]))
    cx, cy = LAYOUT.head_right
    d, w = ts.path_d(ts.text("Convergence", fs["title"], "inter-semibold"), cx, cy)
    out.append(f'<path d="{d}" fill="{th["text"]}"/>')
    out.append(
        fill_path(
            ts.math("f(\\mathbf{x}_k) - f^\\star", fs["formula"]), cx + w + 14, cy, th["text2"]
        )
    )
    return out


def legend_items(runs: list[Run]) -> list[tuple[Run, str]]:
    return [(r, count(r.n_iter) + ("" if r.converged else " (budget)")) for r in runs]


def legend(runs: list[Run], theme: str) -> list[str]:
    th = THEMES[theme]
    fs = FS
    out = []
    x0 = LAYOUT.left[0] if not LAYOUT.legend_rows else LAYOUT.head_left[0]
    x, y = x0, LAYOUT.legend_y
    for run, cnt in legend_items(runs):
        color = series(th, run.cfg)
        out.append(
            f'<path d="M{x} {y - fs["legend"] * 0.33:.1f}h20" stroke="{color}" stroke-width="2.6" stroke-linecap="round"/>'
        )
        if run.cfg.dots:
            out.append(
                f'<circle cx="{x + 10}" cy="{y - fs["legend"] * 0.33:.1f}" r="3.2" fill="{color}" stroke="{th["surface"]}" stroke-width="1.2"/>'
            )
        tx = x + 29
        d, w = ts.path_d(ts.text(run.cfg.label, fs["legend"], "inter-medium"), tx, y)
        out.append(f'<path d="{d}" fill="{th["text"]}"/>')
        if LAYOUT.legend_rows:
            d, w2 = ts.path_d(
                ts.text(cnt, fs["legend"], "inter"),
                W - LAYOUT.head_left[0],
                y,
                anchor="end",
                tnum=True,
            )
            out.append(f'<path d="{d}" fill="{th["text3"]}"/>')
            y += fs["legend"] + 14
        else:
            d, w2 = ts.path_d(ts.text(cnt, fs["legend"], "inter"), tx + w + 7, y, tnum=True)
            out.append(f'<path d="{d}" fill="{th["text3"]}"/>')
            x = tx + w + 7 + w2 + 26
    if LAYOUT.legend_rows:
        d, _ = ts.path_d(
            ts.text("iterations", fs["note"], "inter"),
            W - LAYOUT.head_left[0],
            LAYOUT.legend_y - fs["legend"] - 12,
            anchor="end",
        )
        out.append(f'<path d="{d}" fill="{th["text3"]}"/>')
    return out


def legend_width(runs: list[Run]) -> float:
    w = 0.0
    for run, cnt in legend_items(runs):
        w += 29 + ts.measure(ts.text(run.cfg.label, FS["legend"], "inter-medium")) + 7
        w += ts.measure(ts.text(cnt, FS["legend"], "inter"), tnum=True) + 26
    return w - 26


# ── Assembly ──────────────────────────────────────────────────────────────────────────


def description(runs: list[Run]) -> str:
    """Screen-reader text with the brand's typography (U+2212, thousands separators)."""
    by_speed = sorted(runs, key=lambda r: r.n_iter)
    parts = [
        f"{r.cfg.label} {'converges in' if r.converged else 'stops at the budget after'} {count(r.n_iter)} iterations"
        for r in by_speed
    ]
    return (
        f"Four optimizers on the Rosenbrock function from x₀ = ({num(X0[0], 1)}, {num(X0[1], 0)}). "
        f"Each run stops at the first iterate with ‖∇f(xₖ)‖₂ ≤ 10⁻⁸, within a {count(BUDGET)}-iteration budget: "
        + "; ".join(parts)
        + ". Right: f(xₖ) − f⋆ against k on log–log axes; Newton's curve bends down (quadratic), "
        "BFGS's bends down later (superlinear), the first-order methods fall on straight lines (linear)."
    )


def build(theme: str, *, animated: bool) -> str:
    th = THEMES[theme]
    _p, runs = DATA
    lx, ly, lw, lh = LEFT
    dps = [drawn_path(r) for r in sorted(runs, key=lambda r: PAINT_ORDER.index(r.cfg.id))]
    tag = f"{LAYOUT.name}-{theme}{'-a' if animated else ''}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-labelledby="t-{tag} d-{tag}">',
        f'<title id="t-{tag}">Four optimizers on the Rosenbrock function, computed with numopt</title>',
        f'<desc id="d-{tag}">{description(runs)}</desc>',
    ]
    if animated:
        parts.append(
            "<style>.still{display:none}@media (prefers-reduced-motion: reduce){.anim{display:none}.still{display:inline}}</style>"
        )
    parts.append(
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" fill="{th["surface"]}" stroke="{th["border"]}"/>'
    )
    parts += header(theme)
    clip_id = f"field-{tag}"
    parts.append(
        f'<clipPath id="{clip_id}"><rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="10"/></clipPath>'
    )
    parts.append(f'<g clip-path="url(#{clip_id})">')
    parts += LANDSCAPE[(LAYOUT.name, theme)]
    if animated:
        parts.append('<g class="anim">')
        parts += path_layers(dps, theme, animated=True)
        parts.append('</g><g class="still">')
        parts += path_layers(dps, theme, animated=False)
        parts.append("</g>")
    else:
        parts += path_layers(dps, theme, animated=False)
    parts.append("</g>")
    parts.append(
        f'<rect x="{lx + 0.5}" y="{ly + 0.5}" width="{lw - 1}" height="{lh - 1}" rx="10" fill="none" stroke="{th["border"]}"/>'
    )
    parts += axes_left(theme)
    if animated:
        parts.append('<g class="anim">')
        parts += annotations(runs, theme, animated=True)
        parts += convergence(runs, theme, animated=True)
        parts.append('</g><g class="still">')
        parts += annotations(runs, theme, animated=False)
        parts += convergence(runs, theme, animated=False)
        parts.append("</g>")
    else:
        parts += annotations(runs, theme, animated=False)
        parts += convergence(runs, theme, animated=False)
    parts += legend(runs, theme)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def raster(statics: dict[str, Path], anims: dict[str, Path]) -> None:
    node = shutil.which("node")
    if not node:
        print("node not found: skipping PNG/WebP")
        return
    BUILD.mkdir(parents=True, exist_ok=True)
    render = HERE / "render.mjs"
    for svg in statics.values():
        png = BUILD / svg.with_suffix(".png").name
        subprocess.run([node, str(render), str(svg), str(png), "--scale", "2"], check=True)
        print("wrote", png.relative_to(REPO))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found: skipping WebP")
        return
    for svg in anims.values():
        if "stacked" in svg.name:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            fps = 12
            subprocess.run(
                [
                    node,
                    str(render),
                    "--frames",
                    str(svg),
                    tmp,
                    str(WIDE.W),
                    str(WIDE.H),
                    str(fps),
                    str(T_END + 1.5),
                    "--scale",
                    "1",
                ],
                check=True,
            )
            webp = BUILD / svg.with_suffix(".webp").name
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    f"{tmp}/f%04d.png",
                    "-c:v",
                    "libwebp",
                    "-lossless",
                    "0",
                    "-q:v",
                    "62",
                    "-compression_level",
                    "6",
                    "-loop",
                    "1",
                    str(webp),
                ],
                check=True,
            )
            print("wrote", webp.relative_to(REPO), f"{webp.stat().st_size / 1e6:.2f} MB")


DATA: tuple[object, list[Run]]
LANDSCAPE: dict[tuple[str, str], list[str]] = {}


def main() -> None:
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-raster", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA = run_methods()
    for r in DATA[1]:
        print(f"{r.cfg.id:18s} n_iter={r.n_iter:6d} converged={r.converged} f={r.fs[-1]:.3e}")
    statics, anims = {}, {}
    for lay in (WIDE, STACKED):
        use_layout(lay)
        if not lay.legend_rows:
            lw = legend_width(DATA[1])
            assert lw <= W - 2 * lay.left[0] + 40, f"legend too wide: {lw:.0f}"
        for theme in THEMES:
            LANDSCAPE[(lay.name, theme)] = landscape(DATA[0], theme)
            for animated in (False, True):
                stem = (
                    "hero"
                    + ("-stacked" if lay is STACKED else "")
                    + ("-animated" if animated else "")
                )
                path = OUT / f"{stem}-{theme}.svg"
                path.write_text(build(theme, animated=animated))
                print("wrote", path.relative_to(REPO), f"{path.stat().st_size / 1e3:.0f} kB")
                (anims if animated else statics)[path.name] = path
    use_layout(WIDE)
    if not args.no_raster:
        raster(statics, anims)


if __name__ == "__main__":
    main()
