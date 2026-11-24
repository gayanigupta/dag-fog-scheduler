# Data-Aware DAG Scheduling in Federated Fog Systems

Simulator and experiment code for the paper "Data-Aware Task Scheduling
for Approximate DAG Workflows in Federated Fog Systems".

## Methods

All schedulers take the same instance format and solve the same
assignment + fraction-selection problem.

| Method | File | Description |
|---|---|---|
| MILP | `code/fogsim/milp.py` | Exact mixed-integer model (HiGHS via SciPy); benchmark at moderate sizes |
| LP rounding | `code/fogsim/rounding.py` | Rounded LP-relaxation assignments, feasibility-checked per instance |
| Greedy (data-aware) | `code/fogsim/greedy.py` | `run_greedy`: orders tasks by accuracy density with data-transfer costs |
| Greedy (blind) | `code/fogsim/greedy.py` | `run_greedy` with `data_aware=False`: same rule without bandwidth awareness |
| Greedy (path-shrinking) | `code/fogsim/greedy.py` | `run_greedy_cp`: shrinks the remaining critical path under partial execution |
| Genetic algorithm | `code/fogsim/ga.py` | `run_ga`: permutation GA over task order and node assignment |
| DAG-MEC baseline | `code/fogsim/greedy.py` | `run_dagmec`: adapted from Li et al. (FixDoc placement stage); charges edge transfers at placement, runs every task in full |

## Layout

| Location | Purpose |
|---|---|
| `code/fogsim.py` | CLI entry point (`gen` / `experiment` / `plot`) |
| `code/fogsim/` | Simulator package: `instance.py` (DAG/federation generation), `milp.py`, `rounding.py`, `greedy.py`, `ga.py`, `experiments.py` |
| `code/make_figures.py` | Regenerates the conceptual architecture/DAG diagrams |
| `data/instances/` | Generated workflow instances (JSON) |
| `data/results/` | Experiment CSVs and rendered result figures |
| `notebooks/experiments.ipynb` | Executed notebook that walks through the datasets and reproduces every chart |
| `experiments/` | Standalone experiment drivers (greedy comparisons, seed repeats, upstream-sensitivity, tight-deadline runs) |

## Instance format

Each instance JSON contains:

| Field | Meaning |
|---|---|
| `t` | full execution time of task i on node a (n x m) |
| `A` | per-task accuracy weight (n) |
| `D` | data size of DAG edge (j -> i); 0 if absent (n x n) |
| `B` | bandwidth of link (b, a); diagonal 0 = co-located (m x m) |
| `P` | eligibility: 1 if task i may run on node a (n x m) |
| `t_max` | workflow deadline (= slack factor x optimistic critical path) |

## Reproducing the experiments

Requires Python 3 with NumPy, SciPy, pandas, and matplotlib
(`pip install -r requirements.txt`).

```bash
# generate a workflow instance
python3 code/fogsim.py gen --tasks 50 --nodes 5 --seed 0

# run one experiment -> data/results/<kind>.csv
python3 code/fogsim.py experiment slack     --tasks 30 --nodes 4 --reps 5
python3 code/fogsim.py experiment nodes     --tasks 30 --reps 5
python3 code/fogsim.py experiment size      --nodes 4 --reps 2
python3 code/fogsim.py experiment failure   --tasks 30 --nodes 4 --reps 5
python3 code/fogsim.py experiment bandwidth --tasks 50 --nodes 5 --reps 3

# render figures from the CSVs -> data/results/figures/
python3 code/fogsim.py plot
```

The MILP is skipped automatically for instances above 800 assignment
variables (n*m), matching the paper's use of it as the benchmark at
moderate sizes.
