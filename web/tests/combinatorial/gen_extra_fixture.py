"""Extra Python reference runs for the combinatorial TS port (beyond src/generated/fixtures).

Run from the repo root:
    .venv/bin/python web/tests/combinatorial/gen_extra_fixture.py
It writes web/tests/combinatorial/extra_python.json (test-only data, never bundled).
"""

from __future__ import annotations

import json
from pathlib import Path

from numopt import problems
from numopt.core.registry import run

CASES = [
    ("knapsack_greedy", "knapsack_10", {}),
    ("knapsack_greedy", "knapsack_20", {"single_item_fix": False}),
    ("knapsack_dp", "knapsack_20", {}),
    ("knapsack_branch_bound", "knapsack_20", {"max_nodes": 10}),
    ("knapsack_branch_bound", "knapsack_20", {"record_every": 4}),
    ("knapsack_branch_bound", "knapsack_10", {"max_nodes": 30}),
    ("tsp_nearest_neighbor", "tsp_cities_20", {"start": 7}),
    ("tsp_nearest_neighbor", "tsp_grid_16", {}),
    ("tsp_two_opt", "tsp_cities_20", {"max_iter": 3}),
    ("tsp_two_opt", "tsp_random_15", {"strategy": "best"}),
    ("tsp_two_opt", "tsp_grid_16", {"record_every": 3}),
    ("tsp_or_opt", "tsp_cities_20", {"record_every": 5}),
    ("tsp_or_opt", "tsp_circle_12", {}),
    ("tsp_simulated_annealing", "tsp_circle_12", {"seed": 4, "init": "nearest_neighbor"}),
    ("tsp_simulated_annealing", "tsp_cities_20", {"seed": 0, "max_iter": 3000, "record_every": 250}),
    ("tsp_genetic", "tsp_random_15", {"seed": 5, "pop_size": 10, "crossover_rate": 0.5, "max_iter": 40}),
    ("tsp_genetic", "tsp_cities_20", {"seed": 0}),
    ("tsp_ant_colony", "tsp_random_15", {"seed": 1, "max_iter": 15}),
    ("tsp_ant_colony", "tsp_grid_16", {"seed": 2, "alpha": 0.0, "max_iter": 8}),
    ("tsp_ant_colony", "tsp_circle_12", {"seed": 0, "rho": 1.0, "alpha": 2.0, "max_iter": 6}),
    ("tsp_held_karp", "tsp_random_15", {}),
    ("tsp_held_karp", "tsp_grid_16", {}),
]


def main() -> None:
    out = []
    for method, problem_id, params in CASES:
        seed = params.pop("seed", None)
        kwargs = dict(params)
        if seed is not None:
            kwargs["seed"] = seed
        res = run(method, problems.get(problem_id), **kwargs)
        out.append(
            {"method": method, "problem": problem_id, "params": kwargs, "result": res.to_dict()}
        )
    path = Path(__file__).with_name("extra_python.json")
    path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {len(out)} cases to {path}")


if __name__ == "__main__":
    main()
