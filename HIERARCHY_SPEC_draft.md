# GM-HRL v2: when does the hierarchy matter? (pre-registration DRAFT v2)

Status: DRAFT for author/supervisor review. Once approved, this file, the
environment code and the decision script are committed to the repository
**before** any training run of this study and are not changed afterwards.

## 0. Motivation and what is already known

A diagnostic on the round-2 models (validation instances only, original
benchmark) showed that the Meta-Controller is effectively inert: replacing its
goal by the neutral value (goal attribute 0.5 for every task) leaves
98.6-99.1% of the Sub-Controller's decisions unchanged, and the goal attribute
differs by only 0.015-0.023 (0-1 scale) among candidate tasks; replacing the
goal by random goals changes the makespan by at most 0.5%. Probable reasons:
(i) a benchmark instance is one connected DAG with a uniform SLA threshold, so
there is no workflow-level decision for the upper level; (ii) the
goal-consistency reward has weight 0.1/N; (iii) both levels receive the same
reward; (iv) only the Sub-Controller is warm-started.

This study tests whether a hierarchy helps when there **is** a workflow-level
decision: many concurrent workflows (samples) of different urgency, as in
clinical genomics (STAT vs. routine samples). The round-2 results on the
original benchmark remain in the paper unchanged. This is the second targeted
attempt to find a benefit of the hierarchy (after the capacity-reservation
variant, RESERVATION_SPEC.md); the paper will say so.

## 1. Environment MD2POS-MW (multi-workflow), fixed before training

* A batch consists of K independent workflows (samples). Each workflow is an
  instance of the domain generator with N_w ~ U{10,...,25} tasks (mean 17.5),
  drawn with seeds disjoint from the benchmark; tasks keep their internal
  precedences and RRD branch points.
* Main setting: K = 12 workflows per batch (about 210 tasks).
  Secondary setting (evaluation only, policies of the main setting applied
  without retraining): K = 30 (about 525 tasks).
* Workflow releases: Poisson process, stretched so that the offered load is
  rho = 0.8 (as in the main study). All tasks of a workflow are released with it.
* Urgency: a workflow is STAT with probability s = 0.2 (weight w = 5), routine
  otherwise (w = 1). Secondary settings (evaluation only, main-setting
  policies): s = 0.1 and s = 0.4.
* Deadlines: D_w = r_w + f_class * CP_w, where r_w is the release time and CP_w
  the critical-path length of workflow w under estimated durations at
  submission (without RRD); f_STAT = 2, f_routine = 6 (i.e. the deadline is
  2 or 6 times the critical-path length after release).
* Everything else (resources, capacity, instances per type, duration noise,
  RRD rule) as in Section 5.1 of the paper.
* Splits: test 20 batches x 3 realisations; validation 5 batches x 4
  realisations; training batches freshly generated; seeds disjoint.

## 2. Metrics

* **Primary endpoint:** weighted deadline-violation rate
  WV = sum_w w * 1[C_w > D_w] / sum_w w, where C_w is the completion time of
  the last task of workflow w.
* Secondary (reported, not used for decisions): STAT and routine violation
  rates, mean weighted tardiness, makespan, task-level SLA violation rate,
  utilisation, decision latency.

## 3. Methods (same information and budget for all learning-based methods)

* New node features for **all** learned methods: workflow weight, workflow
  deadline slack (D_w - t)/CP_w clipped to [-3, 3], remaining estimated work
  of the workflow / CP_w. The single-level policy therefore sees exactly the
  information the hierarchy sees; differences come from the decision structure.
* Reward term added to b_k for **all** learned methods (both levels of
  GM-HRL): -(sum of w over workflows whose deadline passes in (t_k, t_k+1]
  before completion) / sum_w w.
* Behaviour-cloning expert for all learned methods: weighted EDF (highest w
  first, then earliest workflow deadline, then highest upward rank).
* GM-HRL v2 (changes fixed in advance):
  - the goal is written per workflow: p_w = sigmoid(<W_g g, W_x x_w> / sqrt(d_g)),
    with x_w the workflow features above; every task of w receives p_w;
  - goal-consistency weight eta_2 = 1.0 (instead of 0.1);
  - during behaviour cloning the Meta-Controller is also trained, by regressing
    p_w onto the expert's workflow ranking (normalised to [0, 1]).
* Baselines: FIFO, HEFT, EDF (earliest workflow deadline), weighted EDF,
  GA (same configuration as in the main study, re-evolved on MD2POS-MW
  training batches with the new features; its fitness curve is reported),
  PPO, GNN-PPO, and GM-HRL (round-2 architecture with the new features, no
  v2 changes).
* Budget: configuration B of the paper for every learned method; 3 seeds.

## 4. Pre-specified checks, hypotheses and decision rule

Validation: genomic and video, main setting; 20 paired runs per domain;
learned methods averaged over 3 seeds; two-sided Wilcoxon signed-rank tests.

* **M1 - is the upper level used?** Share of Sub-Controller decisions that
  change when the goal is replaced by the neutral value. GM-HRL v2 counts as
  non-inert if the share is at least 5%. Rationale: the inert round-2 models
  show 0.9-1.4%; 5% is about 3.5 times that level.
* **M2 - is what it learned informative?** WV of GM-HRL v2 with its own goals
  versus (a) goals replaced by random goals and (b) its goals shuffled across
  workflows, at test time. Informative if own goals give lower WV than both,
  p < 0.05.
* **H1 (primary) - does the hierarchy help?** GM-HRL v2 vs. GNN-PPO on WV:
  relative reduction of at least 5% with p < 0.05, in **both** domains.
* **H2 - does learning beat the rule?** GM-HRL v2 vs. weighted EDF on WV, same
  criterion.

Reporting rule: M1, M2, H1 and H2 are reported whatever their outcome, with
test-set results of all methods, the secondary settings (K = 30; s = 0.1 and
0.4), and GM-HRL v2 on the original benchmark (genomic and video, N = 200).
If H1 holds, the paper states that the hierarchy helps when workflows of
different urgency compete, and reports the settings in which it does and does
not. If H1 fails, the paper states that the hierarchy did not help in this
setting either.

## 5. Not included, with reason

* Separate trainings with a frozen upper or frozen lower level: the
  comparison with GNN-PPO (same encoder, features, reward and budget, no upper
  level) already isolates the contribution of the upper level, and M2
  tests whether the learned goals carry information; three additional
  training conditions would triple the cost without answering a different
  question.
* A makespan constraint in H1: the primary objective of this setting is
  deadline compliance; makespan is reported as a secondary metric.
