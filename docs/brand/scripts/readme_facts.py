"""Live facts for the README, the social card and the portal concept (counts never drift).

    .venv/bin/python docs/brand/scripts/readme_facts.py              # "What's inside" table
    .venv/bin/python docs/brand/scripts/readme_facts.py --research   # "Research" section rows
    .venv/bin/python docs/brand/scripts/readme_facts.py --json       # every fact, machine-readable
    .venv/bin/python docs/brand/scripts/readme_facts.py --portal-js docs/brand/portal/facts.js
    .venv/bin/python docs/brand/scripts/readme_facts.py --write docs/brand/README.draft.md
    .venv/bin/python docs/brand/scripts/readme_facts.py --check docs/brand/README.draft.md

``--write`` replaces both generated tables and the two counts in the prose (the one-liner
"**numopt** — N numerical methods" and "lists the N test problems"); ``--check`` exits with
status 1 and names every stale value (``readme_problems()`` is the same test for pytest).

Every number comes from the registry (`numopt.list_methods()`, `problems.list_problems()`) or from
the files in research/. Raster assets that cannot be regenerated on every release (the social
card PNG uploaded to GitHub's settings) use `facts()["methods_floor"]`, for example "150+", so a
growing registry never makes them wrong.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import numopt
from numopt import problems

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / "research"

# Family id -> (display name, group, problem in TeX, methods to name in the table, in this order).
FAMILY_TEXT = {
    "roots": (
        "Root finding",
        "Equations",
        r"f(x) = 0",
        ["bisection", "newton", "secant", "brent", "itp", "halley"],
    ),
    "systems": (
        "Nonlinear systems",
        "Equations",
        r"F(\mathbf{x}) = 0",
        ["newton_system", "broyden"],
    ),
    "linalg": ("Linear systems", "Equations", r"A\mathbf{x} = \mathbf{b}", []),
    "scalar": (
        "1-D minimization",
        "Optimization",
        r"\min_{a \le x \le b} f(x)",
        ["golden_section", "fibonacci_search", "brent_minimize"],
    ),
    "line_search": (
        "Line search",
        "Optimization",
        r"\min_{\alpha > 0} \varphi(\alpha)",
        ["backtracking", "strong_wolfe", "goldstein"],
    ),
    "unconstrained": (
        "Unconstrained",
        "Optimization",
        r"\min_{\mathbf{x}} f(\mathbf{x})",
        [
            "gradient_descent",
            "nesterov",
            "adam",
            "bfgs",
            "lbfgs",
            "trust_region_steihaug",
            "nelder_mead",
        ],
    ),
    "least_squares": (
        "Nonlinear least squares",
        "Optimization",
        r"\min \tfrac12 \|\mathbf{r}(\mathbf{x})\|_2^2",
        ["gauss_newton", "levenberg_marquardt"],
    ),
    "global": (
        "Global",
        "Optimization",
        r"\min_{\mathbf{x} \in \Omega} f(\mathbf{x})",
        ["cma_es", "differential_evolution", "basin_hopping"],
    ),
    "stochastic": (
        "Stochastic gradients",
        "Optimization",
        r"\min \tfrac1n \textstyle\sum_i f_i(\mathbf{x})",
        ["sgd", "svrg", "saga", "stochastic_adam"],
    ),
    "constrained": (
        "Constrained",
        "Constrained & discrete",
        r"\min f(\mathbf{x}) \;\text{s.t.}\; c(\mathbf{x}) \le 0",
        ["sqp", "augmented_lagrangian", "log_barrier", "frank_wolfe"],
    ),
    "lp": (
        "Linear & integer programming",
        "Constrained & discrete",
        r"\min \mathbf{c}^{\top}\mathbf{x} \;\text{s.t.}\; A\mathbf{x} \le \mathbf{b}",
        ["simplex", "revised_simplex", "primal_dual_ipm", "branch_and_bound"],
    ),
    "combinatorial": (
        "Combinatorial",
        "Constrained & discrete",
        r"\mathbf{x} \in \{0, 1\}^n",
        ["knapsack_dp", "tsp_ant_colony"],
    ),
    "integration": ("Quadrature", "Numerical analysis", r"\int_a^b f(x)\,dx", []),
    "differentiation": (
        "Differentiation",
        "Numerical analysis",
        r"f'(x),\ \nabla f(\mathbf{x})",
        [],
    ),
    "interpolation": ("Interpolation", "Numerical analysis", r"p(x_i) = y_i", []),
    "regression": (
        "Regression",
        "Data",
        r"\min_{\boldsymbol\beta} \sum_i \rho(y_i - \mathbf{a}_i^{\top}\boldsymbol\beta)",
        [],
    ),
}

# One-line findings, quoted from each note's own Discussion (2026-10-05). Automatic extraction of
# the first Discussion sentence was tried and produced fragments, so the line is written by hand:
# when a note's Discussion changes, update its line here. A note without a line is listed as
# "Write-up in progress" (README.md exists) or "Prototype" (code only).
FINDINGS = {
    "adam-successors-rotation": (
        "Re-tuned at every angle, Adam needs 9.2× more iterations on a 45°-misaligned quadratic than "
        "on the aligned one (AdaBelief 5.8×, Lion 5.2×); GD and heavy ball do not change, and Sophia "
        "stays flat (≤ 1.16×)"
    ),
    "anderson-acceleration": (
        "AA(m) reproduces GMRES to 2.5×10⁻¹⁵‖r₀‖ and beats plain fixed-point iteration by 3.3–26×, "
        "but AA(m)-GD needs 1.6–4.3× more gradients than L-BFGS(m) for m ≤ 5"
    ),
    "barycentric-rational-approximation": (
        "AAA converges root-exponentially on √x (fitted C = 4.47 against Stahl's π√2 = 4.44) and "
        "reaches 10⁻¹⁰ on tanh(50x) at degree 20, where Chebyshev interpolation needs 794"
    ),
    "benchmark-profiles": (
        "Derivative-free solver rankings change with the tolerance τ (Kendall τ-b 0.07–0.33); a data "
        "profile read at a small fixed budget ranks solvers differently from a performance profile"
    ),
    "certified-stepsize-schedules": (
        "Every certificate holds on every instance; convex silver steps beat 1/L gradient descent by "
        "45× at n = 4,095 on merely convex problems, but lose to constant 1/L steps on strongly convex ones"
    ),
    "clenshaw-curtis-vs-gauss": (
        "Gauss's factor-2 edge over Clenshaw–Curtis grows with the Bernstein parameter ρ, not with "
        "entireness; with error estimates, nested Clenshaw–Curtis is the cheapest rule on 41–52 % of 27 integrands"
    ),
    "pdlp-restarted-pdhg": (
        "Restarted PDHG converges linearly and solves 29 of 29 LPs to 10⁻⁸ (plain PDHG: 20); against "
        "any one period chosen in advance, adaptive restarts win on 27 of 29, but not against the hindsight-best"
    ),
    "regularized-newton-arc": (
        "Adaptive gradient-regularized Newton converges from all 256 convex starts, where pure Newton "
        "often fails, with fewer Hessians than ARC on 216–246 of them; trust-region Newton is still cheaper"
    ),
    "restarted-accelerated-gradient": (
        "Gradient-restart AGD scales like √κ without knowing μ (slope 0.55) and stays within 1.51× of "
        "tuned Nesterov, but CG and L-BFGS need about 3× fewer iterations on quadratics"
    ),
}


def facts() -> dict:
    by_family: dict[str, list] = defaultdict(list)
    for m in numopt.list_methods():
        by_family[m.family].append(m)
    n = sum(len(v) for v in by_family.values())
    fams = []
    for fam in numopt.FAMILIES:
        ms = by_family.get(fam, [])
        name, group, tex, _pick = FAMILY_TEXT.get(fam, (fam, "Other", "", []))
        fams.append({"id": fam, "name": name, "group": group, "tex": tex, "count": len(ms)})
    return {
        "methods": n,
        "methods_floor": f"{n // 10 * 10}+",
        "families": len(by_family),
        "problems": len(problems.list_problems()),
        "family_rows": fams,
        "research": research_notes(),
    }


def table() -> str:
    by_family: dict[str, list] = defaultdict(list)
    for m in numopt.list_methods():
        by_family[m.family].append(m)
    total = sum(len(v) for v in by_family.values())
    lines = ["| Family | Methods | Examples |", "|:--|--:|:--|"]
    for fam in numopt.FAMILIES:
        ms = by_family.get(fam, [])
        name, _group, _tex, pick = FAMILY_TEXT.get(fam, (fam, "", "", []))
        names = {m.id: m.name for m in ms}
        chosen = [names[i] for i in pick if i in names] or [m.name for m in ms[:4]]
        more = len(ms) - len(chosen)
        ex = ", ".join(chosen[:6]) + (f", +{more} more" if more > 0 else "")
        lines.append(f"| {name} | {len(ms)} | {ex} |")
    lines.append(f"| **Total** | **{total}** | across {len(by_family)} families |")
    return "\n".join(lines)


def _title(note: Path) -> str:
    readme = note / "README.md"
    if readme.exists():
        for line in readme.read_text().splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    method = note / "method.py"
    if method.exists():
        doc = ast.get_docstring(ast.parse(method.read_text())) or ""
        first = doc.strip().split("\n\n")[0].replace("\n", " ")
        first = re.sub(r"``([^`]+)``", r"`\1`", first).rstrip(".")
        if len(first) > 90 and ", " in first[:90]:
            first = first[: first[:90].rindex(", ")]
        return first
    return note.name.replace("-", " ")


def _finding(note: Path) -> str | None:
    """The curated one-line finding (FINDINGS); None when the note has none yet."""
    return FINDINGS.get(note.name)


def research_notes() -> list[dict]:
    if not RESEARCH.is_dir():
        return []
    out = []
    for note in sorted(
        p for p in RESEARCH.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    ):
        has_writeup = (note / "README.md").exists()
        out.append(
            {
                "id": note.name,
                "title": _title(note),
                "finding": _finding(note),
                "status": "write-up" if has_writeup else "prototype",
            }
        )
    return out


def research_md() -> str:
    lines = ["| Note | Finding |", "|:--|:--|"]
    for r in research_notes():
        finding = r["finding"] or (
            "Write-up in progress"
            if r["status"] == "write-up"
            else "Prototype and experiment; write-up pending"
        )
        lines.append(f"| [{r['title']}](research/{r['id']}/) | {finding} |")
    return "\n".join(lines)


# The generated blocks of a README: the marker comment, one blank line, then the table rows.
TABLE_MARK = "<!-- Generated by docs/brand/scripts/readme_facts.py — regenerate"
RESEARCH_MARK = "<!-- Generated by docs/brand/scripts/readme_facts.py --research — regenerate"
# The counts that the prose states (the number is group 2).
ONE_LINER = re.compile(r"(\*\*numopt\*\* — )([\d,]+)( numerical methods)")
PROBLEMS = re.compile(r"(lists the )([\d,]+)(\s+test\s+problems)")


def _block(lines: list[str], mark: str) -> tuple[int, int] | None:
    """[start, end) of the table rows after the marker line, or None without the marker."""
    at = next((i for i, line in enumerate(lines) if line.startswith(mark)), None)
    if at is None:
        return None
    start = at + 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return start, end


def render_readme(text: str) -> str:
    """``text`` with both generated tables and the prose counts replaced by live values."""
    f = facts()
    lines = text.split("\n")
    for mark, body in ((TABLE_MARK, table()), (RESEARCH_MARK, research_md())):
        span = _block(lines, mark)
        if span is not None:
            lines[span[0] : span[1]] = body.split("\n")
    out = "\n".join(lines)
    out = ONE_LINER.sub(lambda m: f"{m[1]}{f['methods']:,}{m[3]}", out)
    return PROBLEMS.sub(lambda m: f"{m[1]}{f['problems']:,}{m[3]}", out)


def readme_problems(path: Path) -> list[str]:
    """Every stale generated value in the README at ``path`` (empty when it is current)."""
    text = path.read_text()
    f = facts()
    lines = text.split("\n")
    out = []
    for mark, body, what in (
        (TABLE_MARK, table(), "the 'What's inside' table"),
        (RESEARCH_MARK, research_md(), "the 'Research' table"),
    ):
        span = _block(lines, mark)
        if span is None:
            out.append(f"{path.name}: no marker for {what}")
        elif "\n".join(lines[span[0] : span[1]]) != body:
            out.append(f"{path.name}: {what} differs from readme_facts.py")
    for rx, want, what in (
        (ONE_LINER, f["methods"], "one-liner method count"),
        (PROBLEMS, f["problems"], "test-problem count"),
    ):
        found = [int(m[2].replace(",", "")) for m in rx.finditer(text)]
        if not found:
            out.append(f"{path.name}: no {what}")
        out += [f"{path.name}: {what} {n}, registry {want}" for n in found if n != want]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--portal-js", type=Path)
    ap.add_argument("--write", type=Path, metavar="README", help="update a README in place")
    ap.add_argument("--check", type=Path, metavar="README", help="exit 1 if a README is stale")
    a = ap.parse_args()
    if a.write:
        a.write.write_text(render_readme(a.write.read_text()))
        print("wrote", a.write)
    elif a.check:
        stale = readme_problems(a.check)
        print("\n".join(stale) if stale else f"{a.check}: current")
        raise SystemExit(1 if stale else 0)
    elif a.portal_js:
        a.portal_js.write_text(
            "// Generated by docs/brand/scripts/readme_facts.py; do not edit.\n"
            f"window.NUMOPT_FACTS = {json.dumps(facts(), ensure_ascii=False, indent=1)};\n"
        )
        print("wrote", a.portal_js)
    elif a.json:
        print(json.dumps(facts(), ensure_ascii=False, indent=1))
    elif a.research:
        print(research_md())
    else:
        print(table())


if __name__ == "__main__":
    main()
