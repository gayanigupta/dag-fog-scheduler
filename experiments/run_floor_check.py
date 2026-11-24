#!/usr/bin/env python3
"""Positive-floor check: what happens when some tasks cannot be skipped?

Every experiment in the paper sets the execution-fraction floor
ell_min = 0, so a task may be skipped entirely.  This script re-runs the
small and medium seeded datasets with a positive floor (0.5) on a
randomly chosen tenth of the tasks and reports score and feasibility
next to the floor-0 runs.

Methods compared: the MILP (per-task floor constraint), the
path-shrinking greedy (per-task floor), and the uniform-fraction greedy
whose single global fraction can only respect the floor globally, so it
runs with ell_min equal to the mandatory floor itself.

Output: experiments/results/floor_check.csv
Run from the repo root:  python3 experiments/run_floor_check.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from fogsim import generate_instance
from fogsim.milp import run_milp
from fogsim.greedy import run_greedy, run_greedy_cp
from fogsim.ga import run_ga

OUT = os.path.join(os.path.dirname(__file__), "results", "floor_check.csv")

DATASETS = [
    ("small",  dict(n_tasks=10,  n_nodes=3, alpha=1.2, density=0.15, seed=0)),
    ("medium", dict(n_tasks=100, n_nodes=5, alpha=1.5, density=0.15, seed=1)),
    ("medium-tight", dict(n_tasks=100, n_nodes=5, alpha=1.0,
                          density=0.15, seed=1)),
]

FLOOR = 0.5
MANDATORY_SHARE = 0.10


def score(inst, ell):
    return float(np.mean(inst.A * ell))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for name, kw in DATASETS:
        inst = generate_instance(**kw)
        rng = np.random.default_rng(kw["seed"] + 7)
        mandatory = rng.random(inst.n) < MANDATORY_SHARE
        floors = {0.0: np.zeros(inst.n),
                  FLOOR: np.where(mandatory, FLOOR, 0.0)}
        print(f"{name}: {int(mandatory.sum())} of {inst.n} tasks mandatory")
        for fl_name, fl in floors.items():
            for meth in ("milp", "greedy_cp", "greedy", "ga"):
                t0 = time.time()
                if meth == "milp":
                    r = run_milp(inst, ell_min=fl)
                elif meth == "greedy_cp":
                    r = run_greedy_cp(inst, ell_min=fl)
                elif meth == "ga":
                    r = run_ga(inst, ell_min=fl)
                else:
                    # uniform greedy: one global fraction, so the floor
                    # is the mandatory level itself
                    r = run_greedy(inst, ell_min=float(fl.max()))
                wall = time.time() - t0
                if r is None:
                    rows.append(dict(dataset=name, floor=fl_name,
                                     method=meth, score=np.nan,
                                     makespan=np.nan, feasible=False,
                                     runtime_s=round(wall, 3)))
                    continue
                assign, ell = r
                mk = inst.forward_pass(assign, ell).max()
                rows.append(dict(dataset=name, floor=fl_name, method=meth,
                                 score=round(score(inst, ell), 4),
                                 makespan=round(float(mk), 2),
                                 feasible=bool(mk <= inst.t_max),
                                 runtime_s=round(wall, 3)))
                print(f"  {name} floor={fl_name} {meth}: score "
                      f"{score(inst, ell):.4f} makespan {mk:.2f} "
                      f"(t_max {inst.t_max})")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(df.to_string())


if __name__ == "__main__":
    main()
