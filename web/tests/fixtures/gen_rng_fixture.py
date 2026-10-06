"""Regenerate tests/fixtures/rng_python.json from numopt.core.rng (run from the repo root):

.venv/bin/python web/tests/fixtures/gen_rng_fixture.py
"""

import json
from pathlib import Path

from numopt.core.rng import Rng

OUT = Path(__file__).with_name("rng_python.json")
cases = []
for seed in (0, 1, 42, 123456789, 4294967295, -7):
    r = Rng(seed)
    case = {"seed": seed, "random": [r.random() for _ in range(1000 if seed == 42 else 200)]}
    r = Rng(seed)
    case["uniform"] = [r.uniform(-3.0, 5.0) for _ in range(50)]
    r = Rng(seed)
    case["normal"] = [r.normal(1.5, 2.0) for _ in range(200)]
    r = Rng(seed)
    case["integers"] = [r.integers(n) for n in list(range(1, 51)) * 2]
    r = Rng(seed)
    case["permutation"] = [r.permutation(n) for n in (1, 2, 5, 10, 31)]
    r = Rng(seed)
    case["choice"] = [r.choice(list("abcdefg")) for _ in range(50)]
    cases.append(case)
OUT.write_text(
    json.dumps(
        {
            "generator": "numopt.core.rng.Rng (Mulberry32)",
            "note": "Generated from Python; do not edit.",
            "cases": cases,
        },
        separators=(",", ":"),
    )
)
print(f"wrote {OUT}")
