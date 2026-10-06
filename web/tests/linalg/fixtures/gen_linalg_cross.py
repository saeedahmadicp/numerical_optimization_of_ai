"""Cross-check fixture for the linalg ports: every linalg method on every linalg problem.

Run from the repo root:  .venv/bin/python web/tests/linalg/fixtures/gen_linalg_cross.py
Writes web/tests/linalg/fixtures/linalg_cross.json (consumed by tests/linalg/cross.test.ts).
Traces are thinned to keep the file small: x, fun and step size of the first 40 and the last 3
steps, and the full info of the first 6 steps and the last step.
"""
import json
from pathlib import Path

import numopt
from numopt import problems
from numopt.core.registry import list_methods
from numopt.core.types import to_jsonable, jsonable_extra

OUT = Path(__file__).with_name("linalg_cross.json")
EXTRA_PARAMS = {
    "sor": [{"omega": 1.0}, {"omega": 1.56}, {"omega": 1.9}, {"omega": 0.6}],
    "gmres": [{"restart": 1}, {"restart": 3}],
    "jacobi": [{"max_iter": 60}],
}
cases = []
for spec in list_methods("linalg"):
    for prob in problems.list_problems("linalg"):
        variants = [{}] + EXTRA_PARAMS.get(spec.id, [])
        starts = [None]
        if prob.A.shape[0] == 2:
            starts += [[-2.0, 2.0], [4.0, 1.5]]
        for params in variants:
            for x0 in starts:
                if x0 is not None and not spec.params:
                    continue
                kw = dict(params)
                if x0 is not None:
                    kw["x0"] = x0
                try:
                    r = numopt.run(spec.id, prob, **kw)
                except (ValueError, TypeError) as e:
                    cases.append({"method": spec.id, "problem": prob.id, "params": kw, "error": str(e)})
                    continue
                tr = []
                last = len(r.trace) - 1
                for i, s in enumerate(r.trace):
                    if 40 <= i < last - 2:
                        continue
                    d = {"k": s.k, "x": s.x, "fun": s.fun, "step_size": s.step_size, "grad_norm": s.grad_norm}
                    if i < 6 or i == last:
                        d["info"] = s.info
                    tr.append(d)
                cases.append(
                    {
                        "method": spec.id,
                        "problem": prob.id,
                        "params": kw,
                        "result": {
                            "x": r.x,
                            "fun": r.fun,
                            "converged": r.converged,
                            "message": r.message,
                            "n_iter": r.n_iter,
                            "extra": jsonable_extra(r.extra),
                            "trace": tr,
                        },
                    }
                )
OUT.write_text(json.dumps(to_jsonable(cases), ensure_ascii=False, separators=(",", ":")))
print(len(cases), "cases,", OUT.stat().st_size // 1024, "KiB")
