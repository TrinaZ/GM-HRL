# GM-HRL: Graph-Mediated Hierarchical Reinforcement Learning for online scheduling with runtime-revealed dependencies

Code and data for the paper

> J. Wang, T. Zheng, X. Sun. *Graph-Mediated Hierarchical Reinforcement Learning with Dynamic Dependency Awareness.* Under review at *Computers & Operations Research*.

The repository contains the simulator, GM-HRL, all baselines and ablations, the offline clairvoyant bound (CP-SAT), the scripts that produce every table in Section 6, the benchmark instances, and the dispatcher for the real-pipeline experiment. `METHOD_SPEC.md` gives the exact specification (state, actions, networks, rewards, training, baselines) that the code implements.

## Repository versions

The paper reports three experimental generations. They share the simulator and the
benchmark; they differ in the meta-controller's goal interface.

| Version | Goal interface | Entry point | Paper |
|---|---|---|---|
| v1.0 | workflow-level target vector | `run_all.sh`, `run_extra.sh` | Tables 1–14, 18 |
| v2 | per-workflow resource share + myopic local view | `run_v2.sh` | Tables 15–17, 19 |
| v3 | per-resource target utilisation $\boldsymbol{r}_t\in[0,1]^4$ | `--variant gmhrl_v3` | Appendix (diagnostic) |

`ASYMMETRIC_SPEC.md` (commit `fe7bf2d`) is the pre-registered protocol for the
asymmetric-observability experiments (v2).

## Layout

```text
gmhrl/
  instance.py    load benchmark instances (with the ai_ft back-edge repair)
  generate.py    deterministic instance generator (recovered seeds of benchmark v1.0)
  sim.py         event-driven simulator: multi-dimensional resources, processor instances,
                 online arrivals, offered load rho, runtime-revealed dependencies (RRD), SLA
  features.py    dynamic dependency graph G_t, node/edge features, global state s_t
  model.py       shared edge-conditioned GNN encoder, Meta-Controller, Sub-Controller, goal writing
  agent.py       rewards (incl. Sim and Frag), rollouts, hierarchical PPO, behaviour-cloning warm start
  baselines.py   FIFO, SJF, online HEFT, Random, WEDF, GA-evolved dispatching rule
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
data/replication/         fresh instances for the replication study (seed 0300–0319),
                          {genomic,video,ai_ft,financial}/N{200,500,1000}/
realrun/                  example pipeline specification and sample sheet
realruns2/                real WGS pipeline runs (3 runs x 2 batches x {fifo,heft,gmhrl})
results/                  per-experiment summary.json + runs.csv (see "Reproducing the paper's results")
fig_timeline.pdf|.png     Fig. 6, realised schedule timeline of the real pipeline
run_all.sh                v1 simulation experiments, restartable by phase
run_extra.sh              v1 single-level GNN-PPO and ORH, then the final evaluation
run_v2.sh                 v2 multi-workflow experiments (asymmetric observability)
InstallationPython $\ge$ 3.10.Bashpip install -r requirements.txt
Training runs on CPU; a GPU gives little benefit because the networks are small and the cost is dominated by per-graph forward and backward passes.Reproducing the experimentsBash# 1. smoke test (a few minutes)
JOBS=2 DOMAINS=genomic SCALES=50 SEEDS=0 \
  TRAIN_COMMON="--bc_iters 2 --critic_warmup 1 --iters 3 --n_train 10" bash run_all.sh ga main eval

# 2. v1 simulation experiments (~3-4 days with JOBS=6 THREADS=2 on a 16-core desktop CPU)
JOBS=6 THREADS=2 bash run_all.sh

# 3. v1 additional baselines and final evaluation (~1-2 days)
JOBS=6 THREADS=2 bash run_extra.sh
# -> results/tables_final.md and results/tables_final.tex

# 4. v2 multi-workflow experiments (asymmetric observability, K=12 and K=30,
#    rho in {0.9, 0.95}, 3 training seeds)
JOBS=6 THREADS=2 bash run_v2.sh
Every phase skips work that is already finished, so an interrupted run can simply be restarted with the same command.Phase (run_all.sh)ContentPapergaGA dispatching rule for every (domain, N)Section 5.2mainGM-HRL and PPO for every (domain, N), 3 training seedsSection 6.1ablationw/o hierarchy, w/o GNN, w/o dynamic graph (genomic, N = 200), 3 seedsSection 6.4pareto$\kappa = 0.3, 3, 10$Section 6.7eval20 test instances $\times$ 3 realisations; CP-SAT clairvoyant bound for $N \le 200$Section 6.1eval_ablationevaluation of ablations and $\kappa$ variantsSections 6.4, 6.7robustness$p_{\text{rrd}}$, $\rho$ and capacity changed at test time, no retrainingSection 6.4transferzero-shot cross-domain and size transfer ($N = 100 \to 200, 500, 1000$)Section 6.2run_extra.sh trains the single-level GNN-PPO baseline and evaluates all methods, including ORH for $N \le 500$, into results/final_<domain>_N<N>, reusing the bounds computed by run_all.sh.run_v2.sh trains the multi-workflow variants (gmhrl_v2, gnn_ppo, wedf, --mw_K 12|30) and evaluates them under symmetric and asymmetric observability.Training time per run2 CPU threads; wall-clock, including behaviour cloning and critic warm-up.ConfigBC + CW + PPON=50N=100N=200N=500N=1000run_all.sh (Tables 8–14)40 + 0 + 30015 min30 min1 h3.5 h8 hrun_v2.sh (Tables 15–17, 19)40 + 10 + 400, --episodes_per_iter 120.8 h1.3 h3.2 h7.9 h—The two rows are different configurations, which is why the N=500 figures differ (3.5 h vs 7.9 h). Table 19 reports the second row.MetricsAll deadline metrics are computed per task and then aggregated over the 20 test instances $\times$ 3 realisations.FieldDefinitionmakespanCompletion time of the last task (schedule length).makespan_over_lbmakespan / LB, where LB is the CP-SAT clairvoyant lower bound (bounds.py).sla_violation_rateFraction of tasks completing after their individual deadline.wvWeighted deadline-violation rate, $\dfrac{\sum_i w_i \cdot \mathbf{1}[C_i > d_i]}{\sum_i w_i}$, with $w_i = 5$ for STAT tasks, $2$ for routine tasks and $1$ otherwise.stat_violSame numerator restricted to STAT tasks, divided by the number of STAT tasks.rout_violSame, restricted to routine tasks.mean_wait / mean_responseMean queueing time / mean completion-minus-release time, in seconds.latency_ms_mean / latency_ms_p95Per-decision wall-clock time (mean / 95th percentile), excluding simulation time.util_cpu … util_diskMean fraction of each resource dimension in use over the schedule horizon.n_rrd_events / n_dynamic_tasksCount of RRD events / tasks revealed at runtime.goal_changedFraction of sub-controller decisions in which the active goal altered the action ranking ($M_1$ diagnostic; null for methods without a goal interface).In summary.json, *_viol and wv are reported as per-run means; paired_tests holds paired Wilcoxon tests against the baseline named in paired_tests_vs. in_domain_paired restricts the comparison to same-domain model-instance pairs (the paper's Tables 15–17); the top-level paired_tests mixes in-domain and cross-domain evaluations and should not be quoted directly.Reproducing the paper's resultsClaimLocationTables 8–10 — main results, $N \in \{50, 100, 200, 500, 1000\}$results/main_{genomic,video,ai_ft,financial}_N*/summary.jsonTable 11 — size transfer ($100 \to 200/500/1000$) & cross-domain transferresults/size_N*/summary.json, results/xdomain_{video,ai_ft,financial}/summary.jsonTable 12 & Fig. 6 — real WGS pipeline (task- and sample-level SLA)realruns2/, fig_timeline.pdfTable 13 — ablations & $\kappa$ trade-offresults/ablation_genomic_N200/summary.jsonTable 14 — robustness ($\rho$, $p_{\text{rrd}}$, capacity)results/robust_{rho0.6,rho0.95,rrd0,rrd2.6667,cap0.7}/summary.jsonTables 15–16 — symmetric observability & secondary scenariosresults/mw_sym_*/summary.json, results/mw_sec_*/summary.jsonTable 17 — asymmetric observability ($K=12, 30$, $\rho \in \{0.8, 0.90, 0.95\}$)results/mw_asym_*/paired_in_domain.jsonTable 18 — per-decision inference latencyresults/final_*_N{100,500,1000}/summary.jsonTable 19 — training cost and sample efficiencyruns_v2/*/train_log.jsonlWEDF and single-level GNN baselinesresults/mw_v3_full_wedf/, results/mw_v2_gnn_baselines/Replication on fresh instancesdata/replication/{domain}/N*/instance_seed03*.jsonTrained checkpointsnot included (model weights are small but numerous); regenerate with bash run_v2.sh or scripts/train.pypaired_in_domain.json is the authoritative source for the paired $p$-values quoted in the paper ($n = 60$ paired runs = 20 instances $\times$ 3 seeds; for $K=30$ weighted violation: genomic $p = 0.0079$, video $p = 0.0078$).Real-pipeline experimentscripts/realrun.py replaces a first-come-first-served dispatcher: it calls the existing stage scripts of a pipeline and only decides which task starts next, subject to declared resource capacities, per-stage concurrency limits and precedence. Policies fifo, heft, orhg and gmhrl use the same interface as in simulation. Every task's release, start, end, exit status, measured peak CPU/RSS and every branch decision are logged, and the metrics and the clairvoyant bound are computed from the log.Bash# dry run with synthetic durations (checks the specification only)
python scripts/realrun.py run --spec realrun/example_wgs.yaml --samples realrun/samples_example.tsv \
  --policy heft --run_dir realruns/dry_heft --dry_run --time_scale 0.005

# real runs, then analysis with a one-hour CP-SAT limit
python scripts/realrun.py run --spec my_pipeline.yaml --samples samples.tsv --policy fifo --run_dir realruns/b1_fifo
python scripts/realrun.py analyze realruns/b1_* --bound --bound_time 3600 --out realruns/summary
v3 Experiment (Resource-Level Hierarchy)The pre-registration protocol, design motivation and pilot results for v3 are documented in PROGRESS.md and HIERARCHY_SPEC_v3.md.
v3 evaluates a high-level abstraction distinct from v1.0/v2: the Meta-Controller outputs per-resource target utilizations $\boldsymbol{r}_t \in [0,1]^4$, and the Sub-Controller logits are penalized by $|\text{util}_{\text{after}} - \boldsymbol{r}_t|$.
The protocol was committed at 5d0ed71 (v3.1) prior to running any training.--variant gmhrl_v3 (goal_mode="none", res_goal=True)New arguments --lambda_res, --eta_cap, --w_r (default = 1.0, fixed)evaluate.py --goal_measure res_neutral (diagnosis for $M_1'$)train_log.jsonl includes the new field train_r_t_range (diagnosis for H3')Bash# pilot run (40 iterations, ~6 minutes on CPU)
python3 scripts/train.py --variant gmhrl_v3 --mw_K 12 --stat_share 0.2 \
  --expert wedf --domains genomic --n_train 20 --iters 40 --episodes_per_iter 4 \
  --bc_iters 10 --critic_warmup 2 --lambda_res 1 --eta_cap 1 --w_r 1 \
  --out runs/v3_pilot_v31 --device cpu

# full training run (400 iterations, ~2.5–3 hours per run); see Section 8 in PROGRESS.md
LicenceCode: MIT (see LICENSE). Benchmark data: CC BY 4.0.
