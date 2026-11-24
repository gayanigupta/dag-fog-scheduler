"""Experiment driver: per-run evaluation, the five experiments of Section VI,
and the result figures.
"""

import os
import time

import numpy as np
import pandas as pd

from . import RES_DIR, FIG_DIR, ROOT
from .instance import generate_instance
from .milp import run_milp
from .rounding import run_rounding
from .greedy import run_greedy, run_greedy_cp, run_dagmec


def run_blind(inst):
    """Bandwidth-blind greedy ablation: the same scheduler and the same
    fraction-shrinking loop as run_greedy, but placements are chosen as
    if transfers were free.  Each pass is checked against the real
    comm-charged makespan (real_deadline=True), so the only difference
    from the data-aware greedy is bandwidth awareness at placement."""
    return run_greedy(inst, data_aware=False, real_deadline=True)
from .ga import run_ga

METHODS = {"milp": run_milp, "rounding": run_rounding,
           "greedy": run_greedy, "greedy_cp": run_greedy_cp,
           "dagmec": run_dagmec, "ga": run_ga}
MILP_CAP = 800          # skip the MILP beyond n*m = 800 (keeps runs bounded)


def evaluate(inst, method, failed=None, seed=0, trials=40):
    t0 = time.time()
    out = run_rounding(inst, trials=trials, seed=seed) \
        if method == "rounding" else \
        (run_blind(inst) if method == "blind" else METHODS[method](inst))
    elapsed = time.time() - t0
    if out is None:
        return dict(score=np.nan, makespan=np.nan, feasible=0,
                    runtime_s=elapsed)
    assign, ell = out
    # retry-once failure model: a failed task occupies its node for twice
    # its execution time but keeps its accuracy when the retry finishes;
    # schedulers then differentiate through deadline feasibility
    y = inst.forward_pass(assign, ell, retry=failed)
    return dict(score=inst.score(ell),
                makespan=float(y.max()),
                feasible=int(y.max() <= inst.t_max * 1.001),
                runtime_s=elapsed)


def experiment(kind, tasks=50, nodes=5, reps=5, seed=0):
    rows = []
    grid = {
        "slack":     [1.0, 1.2, 1.3, 1.5, 2.0],
        "nodes":     [2, 4, 6, 8, 10],
        "size":      [10, 100, 1000],
        "failure":   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "bandwidth": [1.0, 0.5, 0.2, 0.1, 0.05],
    }[kind]
    for x in grid:
        for rep in range(reps):
            s = seed + 1000 * rep
            kw = dict(seed=s)
            if kind == "slack":
                kw.update(n_tasks=tasks, n_nodes=nodes, alpha=x)
            elif kind == "nodes":
                kw.update(n_tasks=tasks, n_nodes=x)
            elif kind == "size":
                kw.update(n_tasks=x, n_nodes=nodes, density=0.05)
            else:
                kw.update(n_tasks=tasks, n_nodes=nodes)
            inst = generate_instance(**kw)
            if kind == "bandwidth":
                # degrade every inter-node link by the same factor -- the
                # deadline stays where it was committed, modeling links
                # deteriorating during an event
                off = ~np.eye(inst.m, dtype=bool)
                inst.B[off] *= x
            failed = None
            if kind == "failure":
                failed = np.random.default_rng(s).random(inst.n) < x
            bound = inst.score(np.ones(inst.n))  # retries preserve accuracy
            # randomized rounding re-solves the LP once per trial, so scale
            # the trial count down on large instances (34 s/solve at
            # n*m = 4000) instead of fixing 40 trials for every size
            trials = max(4, min(40, 20000 // (inst.n * inst.m)))
            # the bandwidth experiment adds the bandwidth-blind baseline so the
            # chart can show what data-aware placement is worth
            meths = list(METHODS) + (["blind"] if kind == "bandwidth" else [])
            for meth in meths:
                if meth == "milp" and inst.n * inst.m > MILP_CAP:
                    rows.append(dict(experiment=kind, param=x, method=meth,
                                     rep=rep, score=np.nan, makespan=np.nan,
                                     feasible=np.nan, runtime_s=np.nan,
                                     bound=bound))
                    continue
                r = evaluate(inst, meth, failed=failed, seed=s + rep,
                             trials=trials)
                rows.append(dict(experiment=kind, param=x, method=meth, rep=rep,
                                 bound=bound, **r))
                print(f"{kind}={x} rep={rep} {meth}: "
                      f"score={r['score']:.3f} feas={r['feasible']} "
                      f"({r['runtime_s']:.1f}s)")
    df = pd.DataFrame(rows)
    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f"{kind}.csv")
    df.to_csv(out, index=False)
    print("wrote", out)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"milp": dict(marker="o", ls="-", color="#0d2c54",
                          label="MILP"),
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
    xlabel = {"slack": r"slack factor $\alpha$",
              "nodes": "fog nodes $m$",
              "size": "workflow size (tasks)",
              "failure": "task failure rate",
              "bandwidth": "link bandwidth scale"}
    os.makedirs(FIG_DIR, exist_ok=True)
    for kind in ["slack", "nodes", "size", "failure", "bandwidth"]:
        csv = os.path.join(RES_DIR, f"{kind}.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        base_csv = os.path.join(ROOT, "revisions", "results",
                                "2026-11-12", "dagmec_experiments.csv")
        if os.path.exists(base_csv):
            bdf = pd.read_csv(base_csv)
            bdf = bdf[bdf["experiment"] == kind][
                ["experiment", "param", "method", "rep", "bound",
                 "score", "makespan", "feasible", "runtime_s"]]
            df = pd.concat([df, bdf], ignore_index=True)

        # ---------- accuracy panel ----------
        fig, ax = plt.subplots(figsize=(3.8, 2.0), dpi=300)
        df_s = df.dropna(subset=["score"])
        xmap = None
        if kind == "bandwidth":
            # even spacing: the interesting scales (0.05-0.2) would
            # otherwise be crammed at the left edge
            ticks = sorted(df_s["param"].unique())
            xmap = {v: i for i, v in enumerate(ticks)}
        if "bound" in df_s.columns:
            bnd = df_s.groupby("param")["bound"].mean()
            xs = [xmap[v] for v in bnd.index] if xmap else bnd.index
            ax.plot(xs, bnd.values, color="#555555", ls="--", lw=1.6,
                    label="Full-execution bound")
        for meth, g in df_s.groupby("method"):
            mean = g.groupby("param")["score"].mean()
            std = g.groupby("param")["score"].std().fillna(0.0)
            xs = [xmap[v] for v in mean.index] if xmap else mean.index
            ax.errorbar(xs, mean.values, yerr=std.values,
                        capsize=3, elinewidth=1.0,
                        **style[meth], lw=2.0, ms=7,
                        mec="white", mew=0.8)
        ax.set_xlabel(xlabel[kind], fontsize=13)
        ax.set_ylabel("accuracy score", fontsize=12.5)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.3, lw=0.5)
        if kind == "size":
            ax.set_xscale("log")
            ax.set_xticks([10, 100, 1000])
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
        elif xmap is not None:
            ax.set_xticks(range(len(ticks)))
            ax.set_xticklabels([f"{v:g}" for v in ticks])
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"perf_{kind}_accuracy.png")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
        plt.close(fig)
        print("wrote", out)

        if kind == "slack":          # one shared legend for the whole grid
            handles, labels = ax.get_legend_handles_labels()
            for m, st in style.items():
                if st["label"] not in labels:
                    ln, = ax.plot([], [], **{k: v for k, v in st.items()
                                             if k != "label"})
                    handles.append(ln); labels.append(st["label"])
            ncols = min(4, len(labels))
            rows = int(np.ceil(len(labels) / ncols))
            figl = plt.figure(figsize=(10.2, 0.42 * rows + 0.15), dpi=300)
            figl.legend(handles, labels, loc="center", ncol=ncols,
                        fontsize=15, frameon=False,
                        columnspacing=1.6, handlelength=2.2)
            figl.savefig(os.path.join(FIG_DIR, "perf_legend.png"),
                         bbox_inches="tight", pad_inches=0.15, facecolor="white")
            plt.close(figl)
            print("wrote legend strip")

        # ---------- second panel (feasibility / runtime) ----------
        if kind == "size":
            fig, ax2 = plt.subplots(figsize=(3.8, 2.0), dpi=300)
            for meth, g in df.groupby("method"):
                rt = g.groupby("param")["runtime_s"].mean()
                ax2.plot(rt.index, rt.values, **style[meth], lw=2.0,
                         ms=7, mec="white", mew=0.8)
            ax2.set_ylabel("mean runtime (s)", fontsize=12.5)
            ax2.set_yscale("log")
            ax2.set_xscale("log")
            ax2.set_xticks([10, 100, 1000])
            ax2.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
            ax2.set_xlabel(xlabel[kind], fontsize=13)
            ax2.tick_params(labelsize=12)
            ax2.grid(alpha=0.3, lw=0.5)
            fig.tight_layout()
            out = os.path.join(FIG_DIR, "perf_size_runtime.png")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
            plt.close(fig)
            print("wrote", out)
        elif kind in ("failure", "bandwidth"):
            fig, ax2 = plt.subplots(figsize=(3.8, 2.0), dpi=300)
            for meth, g in df.groupby("method"):
                feas = g.groupby("param")["feasible"].mean()
                fstd = g.groupby("param")["feasible"].std().fillna(0.0)
                xs = [xmap[v] for v in feas.index] if xmap else feas.index
                ax2.errorbar(xs, feas.values, yerr=fstd.values,
                             capsize=3, elinewidth=1.0, **style[meth],
                             lw=2.0, ms=7, mec="white", mew=0.8)
            ax2.set_ylabel("deadline met", fontsize=12.5)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xlabel(xlabel[kind], fontsize=13)
            ax2.tick_params(labelsize=12)
            ax2.grid(alpha=0.3, lw=0.5)
            if xmap is not None:
                ax2.set_xticks(range(len(ticks)))
                ax2.set_xticklabels([f"{v:g}" for v in ticks])
            fig.tight_layout()
            out = os.path.join(FIG_DIR, f"perf_{kind}_feasibility.png")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
            plt.close(fig)
            print("wrote", out)
