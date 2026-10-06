"""Command-line interface.

numopt list [--family F]                 list methods
numopt problems [--kind K]               list test problems
numopt run METHOD PROBLEM [--x0 ..] [--set k=v ..] [--trace] [--json]
numopt compare PROBLEM METHOD [METHOD ..] [--x0 ..]
numopt export OUT_DIR                    registry + problems + parity fixtures (JSON)
numopt bench PROBLEM [PROBLEM ..] --methods M [M ..] --budget B [--cost C] [--tau T ..]
             [--seeds S ..] [--set M:k=v ..] [--json PATH] [--plot PREFIX]
                                         performance/data profiles (Dolan–Moré, Moré–Wild)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Any

from . import problems as problem_lib
from .core.registry import FAMILIES, get_method, list_methods, run
from .core.types import Result


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        params[key] = _parse_value(value)
    if getattr(args, "x0", None):
        params["x0"] = args.x0 if len(args.x0) > 1 else args.x0[0]
    if getattr(args, "bracket", None):
        params["bracket"] = tuple(args.bracket)
    return params


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.10g}"
    try:
        return "[" + ", ".join(f"{float(t):.10g}" for t in v) + "]"
    except TypeError:
        return str(v)


def _summary_row(r: Result) -> str:
    mark = "✓" if r.converged else "✗"
    return f"{mark} {r.method:<24} iters={r.n_iter:<6} fev={r.n_fev:<6} f={_fmt(r.fun):<18} x={_fmt(r.x)}"


def _bench(args: argparse.Namespace) -> int:
    from . import bench

    params: dict[str, dict[str, Any]] = {}
    for item in args.set or []:
        method, sep1, kv = item.partition(":")
        key, sep2, value = kv.partition("=")
        if not (sep1 and sep2):
            raise SystemExit(f"--set expects METHOD:KEY=VALUE, got {item!r}")
        params.setdefault(method, {})[key] = _parse_value(value)
    res = bench.run_benchmark(
        args.methods,
        args.problems,
        budget=args.budget,
        cost=args.cost,
        params=params,
        seeds=args.seeds,
    )
    width = max(len(s) for s in res.labels)
    print(f"{len(res.instances)} instances, budget {res.budget:g} ({res.cost_model})")
    for tau in args.tau:
        perf = bench.performance_profile(res, tau)
        print(f"\ntau = {tau:g}    {'solved':>7} {'best':>7}   (fractions of instances)")
        for j, label in enumerate(res.labels):
            print(f"  {label:<{width}} {perf.solved[j]:>7.3f} {perf.at(1.0)[j]:>7.3f}")
        if args.plot:
            data = bench.data_profile(res, tau)
            for kind, prof, plot in (
                ("perf", perf, bench.plot_performance_profile),
                ("data", data, bench.plot_data_profile),
            ):
                path = f"{args.plot}_{kind}_tau{tau:g}.svg"
                plot(prof, path)
                print(f"  wrote {path}")
    if args.json:
        print(f"\nwrote {res.save(args.json)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="numopt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list methods")
    pl.add_argument("--family", choices=FAMILIES)

    pp = sub.add_parser("problems", help="list test problems")
    pp.add_argument("--kind", choices=problem_lib.KINDS)

    for name in ("run", "compare"):
        pr = sub.add_parser(name, help=f"{name} method(s) on a problem")
        if name == "run":
            pr.add_argument("method")
            pr.add_argument("problem")
        else:
            pr.add_argument("problem")
            pr.add_argument("methods", nargs="+")
        pr.add_argument("--x0", nargs="+", type=float)
        pr.add_argument("--bracket", nargs=2, type=float)
        pr.add_argument("--set", action="append", metavar="KEY=VALUE", help="method parameter")
        pr.add_argument("--trace", action="store_true", help="print every iteration")
        pr.add_argument("--json", action="store_true", help="print JSON")

    pe = sub.add_parser("export", help="write JSON registry, problems and fixtures")
    pe.add_argument("out_dir")

    pb = sub.add_parser("bench", help="benchmark methods with performance and data profiles")
    pb.add_argument("problems", nargs="+", help="problem ids (each from its default x0)")
    pb.add_argument("--methods", nargs="+", required=True)
    pb.add_argument("--budget", type=float, required=True, help="max cost per run")
    pb.add_argument("--cost", choices=("nfev", "nfev+n*ngev"), default="nfev+n*ngev")
    pb.add_argument("--tau", nargs="+", type=float, default=[1e-1, 1e-3, 1e-5])
    pb.add_argument("--seeds", nargs="+", type=int, default=[0])
    pb.add_argument("--set", action="append", metavar="METHOD:KEY=VALUE", help="method parameter")
    pb.add_argument("--json", metavar="PATH", help="save the benchmark as JSON")
    pb.add_argument("--plot", metavar="PREFIX", help="save PREFIX_{perf,data}_tau*.svg")

    args = p.parse_args(argv)

    if args.cmd == "list":
        for s in list_methods(args.family):
            print(f"{s.family:<16} {s.id:<28} {s.name}  [{s.order}]")
        return 0
    if args.cmd == "problems":
        for pr in problem_lib.list_problems(args.kind):
            print(f"{problem_lib.kind_of(pr.id):<14} {pr.id:<28} {pr.name}")
        return 0
    if args.cmd == "bench":
        return _bench(args)
    if args.cmd == "export":
        from .export import export

        for path in export(args.out_dir):
            print(path)
        return 0

    problem = problem_lib.get(args.problem)
    methods = [args.method] if args.cmd == "run" else args.methods
    params = _params(args)
    results = []
    for m in methods:
        spec = get_method(m)
        allowed = {q.name for q in spec.params} | {"x0", "bracket", "seed"}
        results.append(run(m, problem, **{k: v for k, v in params.items() if k in allowed}))
    if args.json:
        json.dump([r.to_dict(include_trace=args.trace) for r in results], sys.stdout, indent=1)
        print()
        return 0
    for r in results:
        if args.trace:
            print(f"\n{r.method}")
            for s in r.trace:
                print(f"  k={s.k:<5} x={_fmt(s.x):<40} f={_fmt(s.fun)}")
        print(_summary_row(r) + (f"   ({r.message})" if r.message else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
