#!/usr/bin/env python3
"""Upstream-sensitivity sweep (advisor request).

The model scores each task independently: score = mean_i A_i * ell_i.
A reviewer asked how robust the conclusions are if downstream accuracy
instead responds to upstream effort -- e.g. a spread simulation run at
80% effort feeding a routing step that inherits a coarser input.

We cannot profile a real spread->routing chain (no such simulator is in
this repo), so this script presents the sensitivity as a sweep, which is
the option the review allows when the profiling example is skipped.

Propagation rule used for RE-SCORING only (the solvers keep the paper's
objective):

    q_i = A_i * ell_i * (1 - gamma * (1 - mean_in(i)))

where mean_in(i) is the mean fraction of T_i's predecessors and gamma is
the upstream sensitivity.  gamma = 0 recovers the paper's model; gamma =
1 lets a task inherit the full average shortfall of its inputs.

For every (setting, seed, method) we re-run the method on the same
instance family used in the paper, take its returned plan, and re-score
it at gamma in {0, .25, .5, .75, 1}.  Feasibility is unchanged -- the
rule touches accuracy only.  What this can and cannot support:

  CAN:   "conclusions hold (or change) across a range of upstream
          sensitivities"
  CANNOT: "the propagation rule is validated" or "end-to-end accuracy is
          measured" -- it is a re-scoring sensitivity check.

Run from the repo root:  python3 experiments/run_upstream_sensitivity.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from fogsim.instance import generate_instance
from fogsim.experiments import METHODS, run_blind

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
METHODS_ALL = ["milp", "rounding", "greedy", "greedy_cp", "dagmec",
               "ga", "blind"]
# settings where fractions actually shrink (tight + default slack);
# bandwidth 0.2 adds little -- at alpha=1.5 fractions stay near full
SETTINGS = [("slack", 1.0), ("slack", 1.5)]
SEEDS = list(range(10))          # the paper's two five-seed sets, 0-9
OUT = os.path.join(os.path.dirname(__file__),
                   "results", "upstream_sensitivity", "upstream_sensitivity.csv")


def propagated_score(inst, ell, gamma):
    """mean_i A_i*ell_i*(1 - gamma*(1 - mean predecessor fraction))."""
    ell = np.asarray(ell, float)
    q = 0.0
    for i in range(inst.n):
        preds = inst.preds[i]
        factor = 1.0 if not preds else \
            1.0 - gamma * (1.0 - float(np.mean(ell[preds])))
        q += inst.A[i] * ell[i] * factor
    return q / inst.n


def main():
    rows = []
    for kind, param in SETTINGS:
        for seed in SEEDS:
            if kind == "slack":
                inst = generate_instance(alpha=param, seed=seed)
            else:
                inst = generate_instance(seed=seed)
                b = np.isfinite(inst.B)
                inst.B[b] = inst.B[b] * param
            for meth in METHODS_ALL:
                out = (run_blind(inst) if meth == "blind"
                       else METHODS[meth](inst))
                if out is None:
                    continue          # no plan -> nothing to re-score
                assign, ell = out
                y = inst.forward_pass(assign, ell)
                feas = int(y.max() <= inst.t_max * 1.001)
                for g in GAMMAS:
                    rows.append(dict(
                        kind=kind, param=param, seed=seed, method=meth,
                        gamma=g,
                        score=propagated_score(inst, ell, g),
                        feasible=feas))
            print(f"{kind}={param} seed={seed} done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    g = (df.groupby(["kind", "param", "method", "gamma"])
           .agg(score=("score", "mean"), feas=("feasible", "mean"),
                n=("score", "size")).reset_index())
    print("\n=== mean re-scored accuracy (all runs returning a plan) ===")
    print(g.pivot_table(index=["kind", "param", "method"],
                        columns="gamma", values="score")
            .round(3).to_string())
    print("\n=== feasibility rate (independent of gamma) ===")
    print(g.pivot_table(index=["kind", "param", "method"],
                        columns="gamma", values="feas")
            .iloc[:, :1].round(2).to_string())


if __name__ == "__main__":
    main()
