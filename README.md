# GM-HRL: Graph-Mediated Hierarchical Reinforcement Learning for online scheduling with runtime-revealed dependencies

Code and data for the paper

> J. Wang, T. Zheng, X. Sun. *Graph-Mediated Hierarchical Reinforcement Learning with Dynamic Dependency Awareness.* Under review at *Computers & Operations Research*.

The repository contains the simulator, GM-HRL, all baselines and ablations, the offline clairvoyant bound (CP-SAT), the scripts that produce every table in Section 6, and a dispatcher for the real-pipeline experiment. `METHOD_SPEC.md` gives the exact specification (state, actions, networks, rewards, training, baselines) that the code implements.

## Layout

```
gmhrl/
  instance.py    load benchmark instances (with the ai_ft back-edge repair)
  generate.py    deterministic instance generator (recovered seeds of benchmark v1.0)
  sim.py         event-driven simulator: multi-dimensional resources, processor instances,
                 online arrivals, offered load rho, runtime-revealed dependencies (RRD), SLA
  features.py    dynamic dependency graph G_t, node/edge features, global state s_t
  model.py       shared edge-conditioned GNN encoder, Meta-Controller, Sub-Controller, goal writing
  agent.py       rewards (incl. Sim and Frag), rollouts, hierarchical PPO, behaviour-cloning warm start
  baselines.py   FIFO, SJF, online HEFT, Random, GA-evolved dispatching rule
  orhg.py        ORH: online rolling-horizon re-optimisation with CP-SAT
  bounds.py      offline clairvoyant optimum / lower bound with CP-SAT
  realrun/       real-pipeline dispatcher, live state adapter, log analysis
scripts/
  train.py       train GM-HRL, an ablation, single-level GNN-PPO or monolithic PPO
  train_ga.py    evolve the GA baseline
  evaluate.py    paired evaluation on the test instances (+ CP-SAT bound, Wilcoxon tests)
  make_tables.py collect results into the paper's tables (Markdown / LaTeX)
  realrun.py     run / analyse the real-pipeline experiment
  generate_dataset_v1.py  benchmark generator v1.0 (used by gmhrl/generate.py)
data/md2pos_benchmark/    benchmark v1.1, the 400 test instances (see its README)
realrun/                  example pipeline specification and sample sheet
run_all.sh                all simulation experiments, restartable by phase
run_extra.sh              single-level GNN-PPO and ORH, then the final evaluation of all methods
```

## Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

Training runs on CPU; a GPU gives little benefit because the networks are small and the cost is dominated by per-graph forward and backward passes.

## Reproducing the experiments

```bash
# 1. smoke test (a few minutes)
JOBS=2 DOMAINS=genomic SCALES=50 SEEDS=0 \
TRAIN_COMMON="--bc_iters 2 --critic_warmup 1 --iters 3 --n_train 10" bash run_all.sh ga main eval

# 2. all simulation experiments (about 3-4 days with JOBS=6 THREADS=2 on a 16-core desktop CPU)
JOBS=6 THREADS=2 bash run_all.sh

# 3. additional baselines and final evaluation (about 1-2 days)
JOBS=6 THREADS=2 bash run_extra.sh
# -> results/tables_final.md and results/tables_final.tex
```

Every phase skips work that is already finished, so an interrupted run can simply be restarted with the same command.

| Phase (`run_all.sh`) | Content | Paper |
|---|---|---|
| `ga` | GA dispatching rule for every (domain, N) | Section 5.2 |
| `main` | GM-HRL and PPO for every (domain, N), 3 training seeds | Section 6.1 |
| `ablation` | w/o hierarchy, w/o GNN, w/o dynamic graph (genomic, N = 200), 3 seeds | Section 6.4 |
| `pareto` | κ = 0.3, 3, 10 | Section 6.6 |
| `eval` | 20 test instances × 3 realisations; CP-SAT clairvoyant bound for N ≤ 200 | Section 6.1 |
| `eval_ablation` | evaluation of ablations and κ variants | Sections 6.4, 6.6 |
| `robustness` | p_rrd, ρ and capacity changed at test time, no retraining | Section 6.4 |
| `transfer` | zero-shot cross-domain and size transfer (N = 100 → 200, 500, 1000) | Section 6.2 |

`run_extra.sh` trains the single-level GNN-PPO baseline (`train`) and evaluates all methods, including ORH for N ≤ 500, into `results/final_<domain>_N<N>` (`eval`), reusing the bounds computed by `run_all.sh`.

Approximate training time per run (2 CPU threads, 40 behaviour-cloning + 300 PPO iterations): N = 50: 15 min; 100: 30 min; 200: 1 h; 500: 3.5 h; 1000: 8 h.

## Real-pipeline experiment

`scripts/realrun.py` replaces a first-come-first-served dispatcher: it calls the existing stage scripts of a pipeline and only decides which task starts next, subject to declared resource capacities, per-stage concurrency limits and precedence. Policies `fifo`, `heft`, `orhg` and `gmhrl` use the same interface as in simulation. Every task's release, start, end, exit status, measured peak CPU/RSS and every branch decision are logged, and the metrics and the clairvoyant bound are computed from the log.

```bash
# dry run with synthetic durations (checks the specification only)
python scripts/realrun.py run --spec realrun/example_wgs.yaml --samples realrun/samples_example.tsv \
    --policy heft --run_dir realruns/dry_heft --dry_run --time_scale 0.005

# real runs, then analysis with a one-hour CP-SAT limit
python scripts/realrun.py run --spec my_pipeline.yaml --samples samples.tsv --policy fifo --run_dir realruns/b1_fifo
python scripts/realrun.py analyze realruns/b1_* --bound --bound_time 3600 --out realruns/summary
```

## v3 Experiments (Resource-Level Hierarchy)

The pre-registration protocol, design motivation and pilot results for v3 are documented in [`PROGRESS.md`](PROGRESS.md) and [`HIERARCHY_SPEC_v3.md`](HIERARCHY_SPEC_v3.md).
v3 evaluates a high-level abstraction distinct from v1.0/v2: the Meta-Controller outputs per-resource target utilizations \(\boldsymbol{r}_t \in [0,1]^4\), and the Sub-Controller logits are penalized by \(|\text{util}_{\text{after}} - \boldsymbol{r}_t|\).
The protocol was committed at `5d0ed71` (v3.1) prior to running any training.

Key differences of v3 relative to v1.0/v2:

- `--variant gmhrl_v3` (`goal_mode="none"`, `res_goal=True`)
- New arguments `--lambda_res`, `--eta_cap`, `--w_r` (default = 1.0, fixed)
- `evaluate.py --goal_measure res_neutral` (diagnosis for M1')
- `train_log.jsonl` includes the new field `train_r_t_range` (diagnosis for H3')

Reproduction:

```
# pilot run (40 iterations, ~6 minutes on CPU)
python3 scripts/train.py --variant gmhrl_v3 --mw_K 12 --stat_share 0.2 \
  --expert wedf --domains genomic --n_train 20 --iters 40 --episodes_per_iter 4 \
  --bc_iters 10 --critic_warmup 2 --lambda_res 1 --eta_cap 1 --w_r 1 \
  --out runs/v3_pilot_v31 --device cpu

# full training run (400 iterations, ~2.5–3 hours per run)
# see Section 8 in PROGRESS.md
```

## Reproducing the paper's results

| Claim | Location |
|---|---|
| Table 13 (main results, N=200/500) | `results/main_{genomic,video,ai_ft,financial}_N*/summary.json` |
| Table 17 (asymmetric observability, K=12/30) | `results/mw_asym_{s0,K30_s0}/paired_in_domain.json` |
| WEDF comparison | `results/mw_v3_full_wedf/summary.json` |
| Replication on fresh instances | `data/replication/{domain}/N*/instance_seed03*.json` |
| Fig. 6 timeline | `fig_timeline.pdf` |
| Real WGS pipeline | `realruns2/` |
| Trained checkpoints | not included; regenerate with `bash run_v2.sh` |

## Metrics

`wv` = weighted deadline-violation rate (weights 5:2:1 for
STAT : routine : other), normalised by total task count.
`stat_viol` / `rout_viol` = per-class violation rates.
`latency_ms_mean` / `_p95` = per-decision wall-clock time.
## Licence

Code: MIT (see `LICENSE`). Benchmark data: CC BY 4.0.
