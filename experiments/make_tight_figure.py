#!/usr/bin/env python3
"""Tight-deadline figure (paper panel i): score/bound vs slack factor.

Merges alpha_below1.csv (MILP, rounding, greedy, GA) with
greedy_compare.csv (adds the path-shrinking greedy).  Solid lines are
n=10, dashed n=100; thin red rings mark points whose runs were not all
feasible.  Run from the repo root:  python3 experiments/make_tight_figure.py
"""

import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(HERE, "results", "comparisons")
OUT = os.path.join(ROOT, "figures", "fig_tight_deadline.png")

style = {"milp": dict(marker="o", color="#0d2c54", label="MILP"),
         "rounding": dict(marker="s", color="#c0392b",
                          label="Relaxation + rounding"),
         "greedy": dict(marker="^", color="#27ae60", label="Greedy"),
         "ga": dict(marker="D", color="#8e44ad", label="Metaheuristic (GA)"),
         "greedy_cp": dict(marker="X", color="#16a085",
                           label="Greedy (path-shrink)")}

df = pd.concat([pd.read_csv(os.path.join(RES, "alpha_below1.csv")),
                pd.read_csv(os.path.join(RES, "greedy_compare.csv"))],
               ignore_index=True)
df["rel"] = df["score"] / df["bound"]

fig, ax = plt.subplots(figsize=(3.8, 2.0), dpi=300)
ax.axhline(1.0, color="#555555", ls="--", lw=1.6)

for meth in ["milp", "rounding", "greedy", "ga", "greedy_cp"]:
    g = df[df.method == meth]
    if g.empty:
        continue
    for n, ls in [(10, "-"), (100, "--")]:
        gn = g[g.n_tasks == n]
        if gn.empty:
            continue
        mean = gn.groupby("alpha")["rel"].mean()
        feas = gn.groupby("alpha")["feasible"].mean()
        ax.plot(mean.index, mean.values, ls=ls, lw=2.0, ms=6,
                mec="white", mew=0.6, **style[meth])
        bad = feas[feas < 1.0]
        if len(bad):
            ax.plot(bad.index, mean.loc[bad.index].values, "o",
                    ms=11, mfc="none", mec="#d62728", mew=1.3)
ax.set_xlabel(r"slack factor $\alpha$", fontsize=13)
ax.set_ylabel("score / bound", fontsize=13)
ax.tick_params(labelsize=12)
ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print("wrote", OUT)
