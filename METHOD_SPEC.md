# GM-HRL v2 — method and experiment specification

This file states exactly what the code in this repository implements. Every value below is the default in the code; anything changed for a
specific experiment is recorded in that run's `config.json`.

## 1. Benchmark data (dataset v1.1)

* Source: MD2POS benchmark v1.0 (4 domains × N ∈ {50,100,200,500,1000} × 20 instances).
* Generator: `scripts/generate_dataset_v1.py`, wrapped by `gmhrl/generate.py`. The original
  seeds used `hash(domain) % 1000`, which Python randomises per process. The offsets were
  recovered (genomic 851, video 176, ai_ft 496, financial 719) and reproduce the published
  instances exactly (264 instances checked, 0 mismatches).
* Repair: in ai_ft, 86 of 100 instances contain 1–6 edges (u, v) with u > v, i.e. a task
  depending on a later-released task; in 16 instances these close a cycle. They are removed
  (`load_instance(repair_back_edges=True)`).
* Splits: test = the 20 benchmark instances (k = 0..19); validation = generated k = 50..54;
  training = generated k = 100..199. Test instances are never used for training or model
  selection.

## 2. Simulator (`gmhrl/sim.py`)

| Element | Definition |
|---|---|
| Resources | cpu, ram, gpu, disk; capacity 4 units each (1 unit = 8 cores / 32 GB / 1 GPU / 1 disk unit) |
| Processor instances | each task has one eligible type; `m_per_type` = 4 identical instances per type; non-preemptive |
| Online arrivals | task visible at its release date |
| Load ρ | release dates stretched so that bottleneck work / release window = ρ (default 0.8); bottleneck = max over resources of Σ p_i d_ir / R_r and over types of Σ p_i / M_s |
| Durations | actual = estimate × ε, ε ~ LogNormal(−σ²/2, σ), σ = 0.3; revealed at completion |
| RRD | branch point b (rrd_prob > 0) triggers on completion with prob. min(1, m·rrd_prob_b), m = `rrd_mult` (1.0 by default); a trigger inserts 1–2 tasks between b and each original successor; inserted tasks take a successor's type and a duration/demand drawn from that type |
| SLA | Δt_resp = 4 × mean estimated duration of the instance; violation if C_i − r_i > Δt_resp |
| Makespan | max_i C_i − min_i r_i |
| Lower bound LB | submission-time bound: max(release-aware critical path, resource volume, per-type load) |
| Reproducibility | every random outcome is drawn at reset from (instance seed, realisation seed) |

## 3. State, action and graph (`gmhrl/features.py`)

Graph G_t: nodes = visible, unfinished tasks; edges = revealed precedence edges among them,
including RRD-inserted edges.

Node features (26): status one-hot waiting/ready/running/dispatchable (4); p_est / mean p_est
(1); demand cpu/ram/gpu/disk (4); rrd_prob (1); inserted-by-RRD flag (1); elapsed fraction
(1); age / Δt_resp (1); SLA slack (1); upward rank / (10·mean p_est) (1); #successors/5 (1);
#unfinished predecessors/5 (1); **goal attribute written by the Meta-Controller** (1);
processor-type one-hot (8).

Edge features (2): inserted-by-RRD flag; source running.

Global state s_t (23): free capacity / capacity (4); free instances / M_s per type (8);
ready, running, waiting, completed counts / N (4); elapsed time / LB (1); share of tasks not
yet released (1); summed demand of ready tasks / capacity (4); mean age of ready tasks / Δt_resp (1).

Meta-Controller input: [s_t ; z_t] (23 + 64 = 87).

Actions:
* Meta-Controller: g_t ∈ R^8, diagonal Gaussian (initial log-std −1.5), issued every
  Δt_H = 2 × mean estimated duration (simulated seconds).
* Sub-Controller: one dispatchable task (ready, type instance free, demand fits) or WAIT;
  masked softmax; WAIT is masked when no future event exists. Because each task has a single
  eligible type with identical instances, the task–instance pair (i, s, k) reduces to i.

## 4. Networks (`gmhrl/model.py`)

* Shared encoder: input projection 26→64, then L = 2 edge-conditioned message-passing layers
  (width d = 64). MSG is an MLP on [h_u ; h_v ; e_uv], with separate MLPs for messages along
  and against edge direction; mean aggregation over predecessors and successors separately;
  UPDATE: h ← LayerNorm(h + MLP([h ; agg_pred ; agg_succ])).
* Global embedding: z_t = mean pooling of the final node embeddings.
* Goal writing (graph as interface, Figure 2 option A): p_v = σ(⟨W_g g_t, W_x x_v⟩ / √8) is
  written into node attribute 17; the Sub-Controller receives g_t only through the encoder.
* Meta head: MLP(87→128→8) for the mean, value MLP(87→128→1).
* Sub head: task logit MLP([h_v ; c(s_t)]) with c: MLP(23→128→64); WAIT logit MLP([s_t ; z_t])
  (bias initialised to −3); value MLP([s_t ; z_t]).

## 5. Rewards (`gmhrl/agent.py`)

Between consecutive Sub decisions at t_k < t_{k+1}:

* ΔC_k = ΔW / W_ref − Δt / LB (W: summed estimated duration of completed tasks; W_ref: same
  for the N submitted tasks). Summed over an episode it equals W_total / W_ref − makespan / LB.
* U_k = ∫ mean_r used_r / R_r dt / LB
* P_k = (# tasks whose response time exceeded Δt_resp in the interval) / N
* I_k = ∫ mean_r free_r / R_r · 1[some task ready] dt / LB

b_k = w1 ΔC_k + w2 U_k − κ w3 P_k − w4 I_k with (w1, w2, w3, w4) = (1, 0.1, 1, 0.1), κ = 1.

* Meta reward r^M = Σ b_k over its interval.
* Sub reward r^S_k = η1 b_k + (η2 Sim_k − η3 Frag_k)/N with (η1, η2, η3) = (1, 0.1, 0.1).
* Sim_k = p_a / max_{v∈D_k} p_v for the dispatched task a (0 for WAIT).
* Frag_k = mean_r free_r / R_r if tasks are ready but none fits into the free capacity, else 0.

The weights were fixed a priori (primary objectives makespan and SLA at weight 1, shaping
terms at 0.1) and were not tuned on test data.

## 6. Training — H-PPO

* Behaviour-cloning warm start of the Sub-Controller from the HEFT rule: 40 iterations ×
  4 episodes (cross-entropy + value regression).
* PPO: phases alternate every 5 iterations between Sub-Controller (shared encoder, goal writer,
  Sub heads) and Meta-Controller (Meta heads; encoder frozen). Critic-only warm-up: 10 iterations.
* lr 1e-4 (Adam, eps 1e-5); γ_S = 0.995, γ_M = 0.99, GAE λ = 0.95; clip 0.2; entropy 0.01;
  value coefficient 0.5; 4 epochs; minibatch 64; gradient clip 0.5; 4 episodes per iteration;
  300 iterations.
* Model selection: best validation makespan/LB (generated instances k = 50..54, 2 realisations each).
* One policy per (domain, N) unless stated otherwise (see the size-transfer experiment).

## 7. Baselines and ablations

FIFO, SJF, online HEFT (upward rank on the revealed graph), Random, GA-evolved dispatching
rule (weights over 7 features; pop 20, 15 generations, trained on generated instances),
ORHG (rolling-horizon exact re-planning: CP-SAT on the revealed residual instance with
estimated durations, 1 s per solve, re-planned after every RRD event and after arrivals at
most every 0.5 x mean duration; planned start times used as dispatch priorities; HEFT
fallback), single-level GNN-PPO (same encoder, no Meta-Controller; in the style of
Zhang et al., 2020), monolithic PPO (no Meta-Controller, no message passing), and ablations
w/o hierarchy, w/o GNN, w/o dynamic graph (RRD-inserted nodes shown without their edges).
All learning baselines use the same HEFT behaviour-cloning warm start and PPO settings.

## 8. Real-pipeline experiment

`scripts/realrun.py` dispatches the production scripts of each stage under the declared
capacity and per-stage instance limits; policies FIFO (production), HEFT and GM-HRL use the
same interface as in simulation. Every task's release, start, end, exit status, measured peak
CPU/RSS and every branch decision are logged; metrics and the clairvoyant bound are computed
from the log (`realrun/README_realrun.md`).

## 9. Offline clairvoyant bound

For every test (instance, realisation) the realised instance (actual durations, RRD-inserted
tasks and edges) is exported and solved with CP-SAT (interval variables, one cumulative
constraint per resource and per processor type). Reported: best bound, incumbent, status, and
each method's gap to the best bound.
