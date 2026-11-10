# GM-HRL v2: when does the hierarchy matter? Pre-registered protocol (v1.0)

This file, the environment code (`gmhrl/multiwf.py`, the MD2POS-MW parts of
`gmhrl/sim.py`) and the decision script (`scripts/decide_hier.py`) are
committed **before** any learning-based model of this study is trained. They
are not changed afterwards. The draft history is kept in
`HIERARCHY_SPEC_draft.md` and in the git history.

## 0. Motivation

A diagnostic on the round-2 models (validation instances of the original
benchmark only) showed that the Meta-Controller is effectively inert:
replacing its goal by the neutral constant (goal attribute 0.5 for every task)
leaves 98.6-99.1% of the Sub-Controller's decisions unchanged, the goal
attribute differs by only 0.015-0.023 (0-1 scale) among candidate tasks, and
random goals change the makespan by at most 0.5%. Probable reasons: (i) a
benchmark instance is one connected DAG with a uniform SLA threshold, so there
is no workflow-level decision for the upper level; (ii) the goal-consistency
reward has weight 0.1/N; (iii) both levels receive the same reward; (iv) only
the Sub-Controller is warm-started.

This study tests whether a hierarchy helps when there **is** a workflow-level
decision: concurrent workflows (samples) of different urgency, as in clinical
genomics (STAT vs. routine samples). The round-2 results on the original
benchmark remain in the paper unchanged. This is the second targeted attempt
to find a benefit of the hierarchy (after the capacity-reservation variant,
`RESERVATION_SPEC.md`); the paper will say so.

## 1. Environment MD2POS-MW

* A batch consists of K independent workflows. Workflow j of batch k is an
  instance of the domain generator with N_w ~ U{10,...,25} tasks (generator
  seed 100000 + 100k + j); tasks keep their internal precedences and RRD
  branch points (`gmhrl/multiwf.py`).
* Batch indices: test k = 0..19, validation (decision set) k = 50..54, model
  selection during training k = 60..64, training k = 100..199, calibration
  k = 900..939 (calibration batches are used only as described in section 1.1
  and for code smoke tests, section 6).
* Main setting: K = 12 (about 210 tasks). Secondary setting (evaluation only,
  main-setting policies without retraining): K = 30 (about 525 tasks).
* Releases: all tasks of a workflow are released with it; workflow releases
  follow a Poisson process, stretched by the simulator so that the offered
  load is rho = 0.8 (as in the main study).
* Urgency: a workflow is STAT with probability s = 0.2 (weight 5), routine
  otherwise (weight 1). Secondary settings (evaluation only, main-setting
  policies): s = 0.1 and s = 0.4. The STAT draw uses its own random stream,
  so these settings contain the same workflows and releases.
* Deadlines: D_w = r_w + f * CP_w, where r_w is the release time and CP_w the
  critical-path length of workflow w under estimated durations (without RRD),
  with **f = 1.5 for STAT and f = 2 for routine workflows**.
* Everything else (resources, capacity, instances per type, duration noise,
  RRD rule) as in Section 5.1 of the paper.

### 1.1 How the deadline factors were set (recorded as it happened)

1. The draft protocol used f = 2 (STAT) and 6 (routine). Before any learning
   model was trained, the non-learning rules were run on calibration batches
   k = 900..909 (genomic and video, s = 0.2 and 0.4, two realisations each).
   With the draft factors, EDF and weighted EDF missed no deadline at all
   (weighted violation 0%), FIFO 2.5-5.6%; under FIFO the median workflow
   completed within 1.3-1.6 times its critical path. The setting would have
   left nothing to learn.
2. We inspected two further pairs on the same calibration batches:
   (1.25, 2) - weighted EDF still missed about half of the STAT deadlines
   (weighted violation 36-50%), i.e. many deadlines were infeasible - and
   (1.5, 2) - weighted EDF missed 10-20% (genomic 10.1%, video 14.5% at
   s = 0.2). We chose (1.5, 2).
3. A criterion proposed after step 2 (weighted EDF violation in 10-20% and a
   STAT-routine difference of at most 10 points) was checked on fresh
   calibration batches k = 920..939 (s = 0.2): genomic 7.7% (STAT 9.3%,
   routine 4.6%), video 17.2% (STAT 26.0%, routine 9.7%). The criterion was
   not met; the difference from step 2 reflects the small calibration samples
   (10 and 20 batches). Because the criterion was formulated after seeing
   results, we did not tune further: with (1.5, 2) the strongest simple rule
   still misses 8-17% of the weighted deadlines in both calibration sets,
   which leaves room for, but does not favour, any learning method.
4. No learning-based model was trained or evaluated before this choice.

## 2. Metrics

* **Primary endpoint:** weighted deadline-violation rate
  WV = sum_w w * 1[C_w > D_w] / sum_w w, where C_w is the completion time of
  the last task of workflow w.
* Secondary (always reported, not used for decisions): STAT and routine
  violation rates separately, mean weighted tardiness, makespan, task-level
  SLA violation rate, utilisation, decision latency.

## 3. Methods (same information and budget for all learning-based methods)

* New node features for **all** learned methods: workflow weight / 5,
  workflow deadline slack (D_w - t) / CP_w clipped to [-3, 3], remaining
  estimated work of the workflow (summed estimated duration of its unfinished
  visible tasks) / CP_w. The single-level policy therefore sees exactly the
  information the hierarchy sees.
* Reward term added to b_k for **all** learned methods (both levels of
  GM-HRL): -(sum of w over workflows whose deadline passes in (t_k, t_k+1]
  before completion) / sum_w w, weight 1.
* Behaviour-cloning expert for all learned methods: weighted EDF (highest
  weight first, then earliest workflow deadline, then highest upward rank).
* GM-HRL v2 (changes fixed in advance, applied only to GM-HRL v2):
  - the goal is written per workflow: p_w = sigmoid(<W_g g, W_x x_w> / sqrt(d_g)),
    where x_w are the three workflow features; every task of w receives p_w;
  - goal-consistency weight eta_2 = 1.0 (instead of 0.1);
  - during behaviour cloning the Meta-Controller is also trained, by
    regressing p_w onto the expert's workflow ranking (rank among the
    workflows with unfinished visible tasks, scaled to [0, 1], 1 = first).
* Baselines: FIFO, HEFT, EDF (earliest workflow deadline, then upward rank),
  weighted EDF, GA (population, generations and number of training batches as
  in the main study; the three workflow features are appended to its seven
  rule features; fitness = makespan/LB + WV; re-evolved on MD2POS-MW training
  batches; the initial population additionally contains a weighted-EDF-like
  rule, as the learned methods are warm-started from weighted EDF; its fitness
  curve is reported), PPO, GNN-PPO, and GM-HRL with the round-2 architecture
  (new features, reward term and expert, no v2 changes).
* Budget: configuration B of the paper for every learned method (40 BC
  iterations, 10 critic warm-up iterations, 400 PPO iterations x 12 episodes,
  lr 3e-4, entropy 0.02), 100 training batches, 3 seeds.
* Model selection (every learned method): validation WV on the selection
  batches k = 60..64 (realisations 9001, 9002), ties broken by makespan/LB,
  evaluated every 10 iterations (as makespan/LB on k = 50..54 in the main
  study; here a separate set is used so that the decision set k = 50..54 is
  not used for selection).

## 4. Checks, hypotheses and decision rule

Validation: genomic and video, main setting (K = 12, s = 0.2), batches
k = 50..54 x realisations 2000..2003 = 20 paired runs per domain; learned
methods averaged over 3 seeds per run; two-sided Wilcoxon signed-rank tests.
Only the main setting is used for decisions; all secondary settings are
reported descriptively with the same statistics.

* **M1 - is the upper level used?** Share of Sub-Controller decisions (along
  the normal trajectory) that change when the goal attribute is replaced by
  the neutral constant 0.5. Non-inert if >= 5% in both domains. Rationale:
  the inert round-2 models show 0.9-1.4%; 5% is about 3.5 times that.
* **M2 - is what it learned informative?** WV of GM-HRL v2 with its own goals
  versus its goals **shuffled across workflows** at test time (same values,
  same distribution, wrong assignment: at each Meta-Controller decision a
  random order of the workflows is drawn, and at every Sub-Controller decision
  the goal attributes of the workflows present are permuted according to it;
  `_shuffle_goals` in `gmhrl/agent.py`). Informative if own goals give lower
  WV, p < 0.05, in both domains. Random goals (g ~ N(0, I) at each
  Meta-Controller decision, possibly outside the training distribution) are
  reported as a secondary comparison only.
* **H1 (primary) - does the hierarchy help?** GM-HRL v2 vs. GNN-PPO on WV:
  relative reduction >= 5% with p < 0.05, in both domains.
* **H2 - does learning beat the rule?** GM-HRL v2 vs. weighted EDF on WV,
  same criterion.

| M1 | M2 | interpretation |
|----|----|----------------|
| pass | pass | the upper level is used and carries information |
| pass | fail | the lower level uses the goal, but the learned goals carry no information |
| fail | (any) | the lower level ignores the goal; the interface does not work |

Reporting rule: M1, M2, H1 and H2 are reported whatever their outcome,
together with the STAT and routine violation rates of every method, the
test-set results of all methods, and the secondary settings (K = 30;
s = 0.1 and 0.4). If H1 holds, the paper states that the hierarchy helps when
workflows of different urgency compete, in the settings where this was shown.
If H1 fails, the paper states that the hierarchy did not help in this setting
either.

## 5. Not included, with reason

* Separate trainings with a frozen upper or frozen lower level: the comparison
  with GM-HRL's single-level counterpart GNN-PPO (same encoder, features,
  reward, expert and budget) isolates the contribution of the upper level, and
  M2 tests whether its goals carry information.
* A makespan constraint in H1: the objective of this setting is deadline
  compliance; makespan is reported as a secondary metric.
* GM-HRL v2 on the original benchmark: its goal is defined per workflow, and
  an original benchmark instance is a single connected workflow, so the v2
  goal would be one constant per instance.

## 6. Code tests before this commit (disclosed)

The code was smoke-tested before this file was committed: training runs of at
most 3 iterations on at most 4 training batches, a GA run with population 4
and one generation, and the complete pipeline (`run_hier.sh`) with such
budgets on calibration batches k = 900..905. One early smoke evaluation used
validation batches k = 50, 51 (2 realisations) with models trained for at most
3 iterations; it only checked that the evaluation code runs and is not used
for anything. No model trained with the budget of section 3 existed when this
protocol was committed.

## 7. Files

* `gmhrl/multiwf.py` (batch generator), `gmhrl/sim.py` (deadlines and workflow
  metrics), `gmhrl/features.py` (workflow features), `gmhrl/model.py`
  (`goal_mode="workflow"`), `gmhrl/agent.py` (deadline reward term, meta
  behaviour cloning, goal diagnostics), `gmhrl/baselines.py` (EDF, weighted
  EDF, GA features), `scripts/train.py` (`--mw_K`, `--expert wedf`, variant
  `gmhrl_v2`), `scripts/train_ga.py`, `scripts/evaluate.py` (`--mw_K`,
  `--goal_measure`, `NAME=model.pt@shuffle|random|neutral`),
  `scripts/decide_hier.py` (section 4), `run_hier.sh` (all phases).
