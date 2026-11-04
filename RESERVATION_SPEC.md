# GM-HRL+R: capacity reservation for runtime-revealed work

Pre-registration. This file, `gmhrl/reserve.py` and `scripts/decide_reserve.py`
are committed **before** any GM-HRL+R validation run. Neither the mechanism,
its constants, the validation protocol nor the adoption rule are changed
after validation results are seen.

## 1. Motivation

Section 4 of the manuscript claimed that the Meta-Controller can reserve
capacity for downstream tasks that are not yet revealed. The implementation
used for the round-2 results had no such mechanism. GM-HRL+R implements it.
The existing GM-HRL models (round 2, `runs_v2/gmhrl_*`) are the "without
reservation" control and are not retrained.

## 2. Mechanism (implemented in `gmhrl/reserve.py`)

* **Branch points.** An original task with `rrd_prob > 0`. While a branch
  point runs, the work it may reveal is anticipated.
* **Online statistics.** Computed from the revealed state only, never from
  hidden realisations, separately for each processor type k of branch points:
  `n_k` completed branch points, `t_k` of them revealed at least one task;
  branching probability `p_k = t_k / (n_k + 5)` (5 pseudo-observations
  without a trigger, i.e. a conservative cold start that shrinks towards 0);
  mean revealed demand `d_k` = mean over the triggering branch points of the
  summed demand of the tasks they revealed, `d_k = 0` until the first trigger
  of type k (no reservation before any evidence).
* **Reservation level.** The Meta-Controller's Gaussian goal has one extra
  dimension; `g_res = sigmoid(g[d_g])` in [0, 1]. The level in force is the
  one written at the latest Meta-Controller decision (0 before the first).
  The bias of this output is initialised to -1 (g_res about 0.27).
* **Reserved capacity.**
  `R_raw = g_res * sum over running branch points u of p_type(u) * d_type(u)`,
  `R = min(R_raw + hand-offs, 0.25 * capacity, free capacity)` elementwise.
  Concurrent branch points contribute separately; their total is bounded by
  the two caps; each branch point's share is its fraction of R_raw.
* **Release.** When a branch point completes, its share is released. If it
  revealed tasks, the share is kept only for the decisions taken at its
  completion time (the revealed tasks become ready in that decision epoch) and
  released as soon as time advances.
* **Dispatch rule.** Revealed tasks may use all free capacity. An ordinary
  task may start only if its demand fits into `free - R`, except
  (a) **anti-starvation**: a task that has been ready for longer than the
  SLA threshold ignores the reservation, and (b) **liveness**: if no task
  could start and no future event exists, the reservation is ignored.
  Processor-instance slots are not reserved.
* **Scope.** Only the hierarchical model reserves (the reservation is a
  Meta-Controller decision). Baselines (FIFO, SJF, HEFT, GA, ORH, PPO,
  GNN-PPO) are unchanged.
* **Logging.** Per decision with R > 0: time, g_res, R_raw, R, number of
  running branch points, hand-offs, ordinary tasks blocked; every release
  (branch point, amount); every use of reserved capacity by a revealed task
  (task, amount). Summaries per run: share of decisions with R > 0, mean
  reserved fraction, blocked (task, decision) pairs and distinct blocked
  tasks, used and released totals, anti-starvation and liveness bypasses.
  Simulator: `evaluate.py --res_log` writes `<out>/res_logs/*.jsonl`. Real
  pipeline: the dispatcher writes `reservation.jsonl` and a `reservation`
  event in `events.jsonl`.

## 3. Training

Variant `gmhrl_res` (`scripts/train.py`), identical to GM-HRL otherwise, with
the round-2 budget (configuration B): 40 BC iterations, 10 critic warm-up
iterations, 400 PPO iterations x 12 episodes, lr 3e-4, entropy 0.02, model
selection on validation makespan as for every other method.

## 4. Validation protocol (decides adoption)

* Domains genomic and video, N = 200; models: `gmhrl_res` seeds 0, 1, 2
  (new) and `gmhrl` seeds 0, 1, 2 (round 2, unchanged).
* Validation instances k = 50..54 (generated, never used for testing),
  realisation seeds 2000..2003: 20 paired runs per domain. Each run's value
  is averaged over the three training seeds of a method.
* Statistics per domain: `dmk` = mean over the 20 pairs of
  (makespan_R / makespan_0 - 1); `dsla` = mean SLA violation rate of R divided
  by that of the control, minus 1; p-values from two-sided Wilcoxon
  signed-rank tests on the 20 paired makespans and SLA violation rates.
* **Criterion (same as the round-2 selection criterion).** A domain passes if
  (i) `dmk <= -1%` and `p_mk < 0.05`, or
  (ii) `dmk <= +0.5%` (makespan not worse by more than 0.5%) and
  `dsla <= -5%` (relative) and `p_sla < 0.05`.
* **Adoption rule.** GM-HRL+R is adopted as the method of the paper if and
  only if **both** domains pass. `scripts/decide_reserve.py` prints and saves
  the decision; it is not overridden.

## 5. Consequences

* **Adopted.** GM-HRL+R replaces GM-HRL as the method of the paper: it is
  trained for all domains and N in {50, 100, 200, 500} (3 seeds; N = 1000
  zero-shot from N = 500), for the ablations w/o GNN and w/o dynamic graph,
  and for the kappa variants; all GM-HRL evaluations (main, ablation,
  robustness, transfer, latency) are repeated. The round-2 GM-HRL becomes the
  ablation "w/o reservation". The real pipeline is run twice more with
  GM-HRL+R (genomic, N = 500, seed 0, fixed in advance); FIFO, HEFT and
  GM-HRL runs are kept.
* **Not adopted.** The paper reports the validation result ("reservation did
  not improve on GM-HRL under the pre-specified criterion") and the test-set
  results of the six GM-HRL+R models at genomic and video N = 200, and the
  claim about proactive reservation is removed from Section 4.
* In both cases, the test-set results of the six N = 200 models are reported.
