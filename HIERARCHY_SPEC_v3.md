# GM-HRL v3: resource-level hierarchy. Pre-registered protocol (v3.0)

* Parent protocol: `HIERARCHY_SPEC.md` v1.0, commit `4fec357` (2026-11-11
  04:31 +0800). v1.0 and its amendment v1.1 are not changed by this file.
* Date: 2026-11-25. The commit time and hash of this file are recorded by
  git and GitHub; they are not written into this file.
* Scripts: `run_hier_v3.sh` (training), `run_hier_v3_eval.sh` (evaluation).

## 0. Motivation

v1.0 showed that a workflow-level goal is inert: M1 = 1.9-2.8% (below the 5%
threshold), M2/H1/H2 not met. The round-2 diagnostic had already shown that
the Meta-Controller of the original benchmark is inert (98.6-99.1% of Sub
decisions unchanged under a neutral goal). Diagnosis (v1.0 Section 0 and
Section 6.4): the Sub-Controller receives the three workflow features as node
features (v1.0 Section 3), so a workflow-level goal carries no information it
does not already have. The interface is redundant, not lossy.

v3 tests a different high-level abstraction: a resource-level goal
r_t = [r_cpu, r_ram, r_gpu, r_disk] in [0,1]^4, the target utilisation of
each resource over the next Delta t_H. This is forward-looking and aggregated
over a time window. The Sub-Controller's per-event observation contains the
current free capacity and the summed demand of the ready tasks
(`features.py`, global state), but not a target utilisation over the next
window. The interface therefore carries information the lower level cannot
infer from its own observation.

This is the third targeted attempt to find a benefit of the hierarchy (after
the capacity-reservation variant, `RESERVATION_SPEC.md`, and the
workflow-goal variant, `HIERARCHY_SPEC.md` v1.0). The paper will say so.

## 1. Environment, unchanged

MD2POS-MW exactly as in `HIERARCHY_SPEC.md` v1.0: K independent workflows per
batch, Poisson releases stretched to rho = 0.8, STAT share s = 0.2 (weight 5),
deadline factors f = 1.5 (STAT) and 2 (routine), 4 resource dimensions
(cpu, ram, gpu, disk), 4 instances per processor type. No change to
`gmhrl/sim.py`, `gmhrl/multiwf.py` or the instance generator.

**Known limitation (stated, not worked around).** Task demands are fixed
vectors, not stage-dependent, and GPU memory and network I/O are not modelled
as separate dimensions. Stage-dependent demands (e.g. BWA: I/O -> CPU -> I/O)
and GPU-memory pressure (AI fine-tuning) are future work; v3 tests whether
the resource-level goal carries information under the fixed-demand model.

## 2. What v3 changes relative to v1.0 (exactly three items)

(a) **Meta action.** The Meta-Controller outputs r_t in [0,1]^4 (target
    utilisation per resource), issued every Delta t_H = 2 * mean p_est. The
    mapping from the Meta network output g in R^{d_g} to r_t is
    r_t = sigmoid(W_r g + b_r), with W_r in R^{4 x d_g}, b_r in R^4,
    initialised (W_r = 0.1 * default, b_r = 2) so that r_t is close to 1 at
    the start (no cap). The workflow goal p_w of v1.0 is not used
    (`goal_mode="none"`); the workflow features remain as node features for
    all learned methods, as in v1.0 Section 3.

(b) **Sub action.** Unchanged (task choice + wait), but the logit of task i
    is penalised by
        -lambda_res * sum_k |util_after(i)_k - r_t_k|,
    where util_after(i) = (cap - free + dem[i]) / cap in [0,1]^4. r_t is the
    target utilisation, not an upper bound: the Sub-Controller is penalised
    both when it exceeds and when it falls below r_t.

(c) **Rewards.** Two additions, both present for all learned methods:
    - Sub reward: the goal-consistency term Sim_k of v1.0 is replaced by
        -eta_cap * mean_k (util_mean_k - r_t_k)^2,
      where util_mean is the average utilisation of the interval since the
      last Sub decision.
    - Meta reward: b_k gains -w_r * mean_k (util_mean_k - r_t_k)^2, with
      util_mean averaged over the current Meta interval. `res_head` is
      trained by this term only: it is part of the Meta optimiser, and the
      Sub penalty above uses the detached r_t, so the two levels are updated
      by separate gradients.
    All other reward terms (w1 dC, w2 U, w3 P, w4 I, w6 M) are as in v1.0.

No other change: encoder, GNN, PPO hyperparameters, training budget,
behaviour-cloning expert (weighted EDF), seeds and evaluation instances are
those of v1.0.

## 3. Hyperparameter values (fixed before running)

The three new hyperparameters take their **default values**
    lambda_res = 1.0,  eta_cap = 1.0,  w_r = 1.0
in the pilot and in the main runs. No grid search over these values is
performed on the validation set, to avoid overfitting to 20 validation runs.

**One pre-registered exception.** A 40-iteration pilot on the training
batches (k = 100..119, genomic, s = 0.2, rho = 0.8) is run before the main
runs. If at iteration 40 the training metric `r_t_range` (max over episodes
and resources of max(r_t) - min(r_t), computed by `agent.py`) is below 0.05,
the pilot is repeated once with w_r = 10.0, all other values unchanged.
If that pilot also gives r_t_range < 0.05, the main runs are executed with
the default values anyway, and the negative result is reported with the H3'
diagnostic (Section 4). No other hyperparameter values are tried.

## 4. Checks, hypotheses and decision rule

Validation: genomic and video, main setting (K = 12, s = 0.2, rho = 0.8),
batches k = 50..54 x realisations 2000..2003 = 20 paired runs per domain;
learned methods averaged over 3 seeds; two-sided Wilcoxon signed-rank tests.
Only the main setting is used for decisions; s = 0.1, 0.4, K = 30 and the
load sweep rho = 0.7, 0.9, 0.95 are descriptive.

* **M1' - is the resource goal used?** Share of Sub-Controller decisions
  that change when r_t is replaced by the neutral value (1,1,1,1) at every
  Meta-Controller decision, measured on the validation batches. Threshold:
  >= 5% in both domains (same rationale as v1.0 M1, 3.5x the 1.4% of the
  inert round-2 model).
* **H3' - did the Meta-Controller learn a non-trivial goal?** For each
  validation episode, compute range_k = max_t r_t,k - min_t r_t,k for each
  resource k over the episode. H3' holds if max_k range_k > 0.05 in at least
  50% of validation episodes, in both domains. This separates "Meta learned
  nothing" (H3' fails) from "Meta learned but Sub ignored it" (H3' passes,
  M1' fails).
* **H1' (primary) - does the hierarchy help?** GM-HRL v3 vs GNN-PPO on WV:
  relative reduction >= 5% with p < 0.05, in both domains.
* **H2' - does learning beat the rule?** GM-HRL v3 vs weighted EDF on WV,
  same criterion.
* **H4' (descriptive, no threshold) - does the resource-level goal beat the
  workflow-level goal?** GM-HRL v3 vs GM-HRL v2 on WV, on the same batches.
* **H5' (descriptive, no threshold) - zero-shot transfer across domains.**
  The genomic and video v3 policies applied, without retraining, to the AI
  fine-tuning and financial domains (K = 12, s = 0.2, rho = 0.8). Reported
  as a table of WV and of the mean r_t per resource.

| M1' | H3' | interpretation |
|-----|-----|----------------|
| pass | pass | the interface is used and the goal is non-trivial |
| fail | pass | the Meta-Controller learned a goal, the Sub-Controller ignored it |
| (any) | fail | the Meta-Controller did not learn a non-trivial goal; H1'/H2' would not test the interface |

Reporting rule: M1', H3', H1' and H2' are reported whatever their outcome,
together with the STAT and routine violation rates of every method, the
test-set results of all methods, and the secondary settings and the
cross-domain transfer. If H1' holds, the paper states that the resource-level
hierarchy helps when workflows of different urgency compete, in the settings
where this was shown. If H1' fails, the paper states that the hierarchy did
not help in this setting either, with the M1'/H3' diagnosis.

## 5. Stop rule (fixed before running)

If H1' and H2' are both not met on validation, no further tuning is done:
the test-set results are reported as they are, and the paper states the
negative result with the M1'/H3' diagnosis. If H3' fails, the paper states
that the resource-level goal was not learned, and reports H1'/H2' as not
interpretable. In both cases the paper does not claim that resource-level
hierarchy helps.

## 6. Settings

Main: K = 12, s = 0.2, rho = 0.8, genomic and video, 3 training seeds.
Secondary (descriptive): s = 0.1, 0.4; K = 30; rho = 0.7, 0.9, 0.95.
Cross-domain (descriptive): policies trained on genomic/video, evaluated on
AI fine-tuning and financial.
Instances: test k = 0..19, realisations 1000..1002; validation k = 50..54,
realisations 2000..2003; model selection k = 60..64, realisations 9001,
9002; training k = 100..199.
Budget: configuration B of the paper (40 BC + 10 critic warm-up + 400 PPO
iterations x 12 episodes, lr 3e-4, entropy 0.02), 100 training batches,
3 seeds.

## 7. What was known when this spec was written

At the time of this commit:
* v1.0 results (M1 = 1.9-2.8%, M2/H1/H2 not met) and the round-2 diagnosis
  (Meta inert, 98.6-99.1% of decisions unchanged under a neutral goal) are
  known.
* The capacity-reservation variant (`RESERVATION_SPEC.md`) was not adopted
  (genomic +6.4% relative SLA reduction but video +1.6% makespan and +5.7%
  SLA; the adoption rule was not met).
* The amendment v1.1 load-sensitivity sweep has been run for v1.0 and v2
  only; no result of v3 exists at any load.
* A v3 smoke test (2 iterations, 4 training batches, 2 episodes per
  iteration) was run before this commit; it produced no error and no NaN,
  and its r_t_range (~0.006) is the expected value at this budget, because
  `res_head` is only trained in the Meta phase (iterations 5-9 of each
  10-iteration cycle), which the 2-iteration smoke test does not reach.
  No result of a 40-iteration pilot exists at the time of this commit.

## 8. Reporting

All v3 results are reported in a new Section 6.5 whatever their direction,
including M1' and H3' diagnostics and the descriptive H4'/H5' tables. If
H1'/H2' fail, the paper reports the negative result and the M1'/H3'
diagnosis, and does not claim that resource-level hierarchy helps.

## 9. Files

* `gmhrl/model.py`: `GMHRLNet` gains `res_goal` / `n_resource` arguments,
  a `goal_mode="none"` branch in `write_goal`, a `res_head` (Linear(d_g, 4),
  bias 2.0, weight * 0.1) and `resource_goal(g)`.
* `gmhrl/agent.py`: `res_head` is in the Meta optimiser; `_res_penalty`
  uses `|util_after - r_t|`; `_base_reward` gains the `w_r` term on the
  interval average `util_mean`; `rollout` records `g_det`, `util_int_accum`,
  `dt_accum` and `util_mean` per Meta transition; `update(phase="meta")`
  adds the `w_r` loss on the detached `g`.
* `gmhrl/features.py`, `gmhrl/baselines.py`, `gmhrl/sim.py`,
  `gmhrl/multiwf.py`: no change.
* `scripts/train.py`: variant `gmhrl_v3`, flags `--lambda_res`,
  `--eta_cap`, `--w_r`.
* `scripts/evaluate.py`: `--goal_measure res_neutral` for M1' and H3'.
* `run_hier_v3.sh`, `run_hier_v3_eval.sh`.

## 10. Code tests before this commit (disclosed)

The code was smoke-tested before this file was committed: a 2-iteration
training run (`--n_train 4 --iters 2 --episodes_per_iter 2 --bc_iters 2
--critic_warmup 1 --lambda_res 1 --eta_cap 1 --w_r 1`) on genomic,
K = 12, s = 0.2, rho = 0.8. It completed without error or NaN. Its
`r_t_range` (~0.006) is the expected value at 2 iterations, because
`res_head` is only trained in the Meta phase (Section 2(c)). No pilot and
no main run existed when this protocol was committed.