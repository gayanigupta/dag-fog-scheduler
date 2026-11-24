#!/usr/bin/env python3
"""Panel: re-scored task accuracy vs upstream sensitivity gamma (alpha=1).

Reads experiments/results/upstream_sensitivity/upstream_sensitivity.csv and writes
figures/perf_upstream_sensitivity.png in the same style as the
other Fig. 3 panels.  Only slack alpha=1.0 is shown: at alpha=1.5 every
method sits at the full-execution bound, so the sweep is flat.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "experiments", "results", "upstream_sensitivity",
                   "upstream_sensitivity.csv")
OUT = os.path.join(ROOT, "figures",
                   "perf_upstream_sensitivity.png")

style = {"milp": dict(marker="o", ls="-", color="#0d2c54", label="MILP"),
         "rounding": dict(marker="s", ls="--", color="#c0392b",
                          label="Relaxation + rounding"),
         "greedy": dict(marker="^", ls=":", color="#27ae60",
                        label="Greedy"),
         "ga": dict(marker="D", ls="-.", color="#8e44ad",
                    label="Metaheuristic (GA)"),
         "blind": dict(marker="v", ls=":", color="#7f8c8d",
                       label="Greedy (bandwidth-blind)"),
         "dagmec": dict(marker="P", ls="-", color="#d35400",
                        label="DAG-MEC baseline"),
         "greedy_cp": dict(marker="X", ls="-", color="#16a085",
                         label="Greedy (path-shrink)")}

df = pd.read_csv(CSV)
df = df[(df.kind == "slack") & (df.param == 1.0)]

fig, ax = plt.subplots(figsize=(3.4, 2.3))
for meth, g in df.groupby("method"):
    mean = g.groupby("gamma")["score"].mean()
    ax.plot(mean.index, mean.values, lw=2.0, ms=7, mec="white", mew=0.8,
            **style[meth])
ax.set_xlabel(r"upstream sensitivity $\gamma$", fontsize=13)
ax.set_ylabel("accuracy score", fontsize=12.5)
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15,
            facecolor="white", dpi=200)
print("wrote", OUT)
