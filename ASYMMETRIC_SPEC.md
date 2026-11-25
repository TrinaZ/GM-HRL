# Pre-Registered Protocol: Asymmetric Observability in GM-HRL (v4.0)

## 1. Motivation and Hypothesis
In centralized full-observability experiments (`4fec357`, `5d0ed71`), the Sub-Controller receives both global system congestion state (`s_t`, `z_t`) and workflow-level urgency/slack attributes (`WF_IDX:WF_IDX+WF_DIM`), rendering top-down hierarchical goals redundant. In distributed or privacy-isolated multi-tenant scheduling, local dispatchers observe only task-local attributes and immediate readiness, whereas the global coordinator observes the full dependency graph.

We test whether **information asymmetry** restores the functional necessity of the graph-mediated hierarchical interface (`--sub_obs myopic`):
- **Sub-Controller (`sub_obs="myopic"`)**: `WF_IDX:WF_IDX+WF_DIM` is zeroed out prior to the GNN encoder, and `s_t` / `z_t` are masked to zero in `sub_logits`. The Sub-Controller perceives workflow priority exclusively via the Meta-Controller's goal attribute `GOAL_IDX`.
- **Meta-Controller**: Retains full observation `(s_t, z_t)` and full workflow features `WF_IDX:WF_IDX+WF_DIM` for goal writing (`goal_mode="workflow"`).

## 2. Pre-Specified Criteria (Evaluated on 20 Test Instances x 3 Realisations, n=60/domain)
- **M1_asym (Goal Utilisation)**: Replacing learned workflow goals with neutral `0.5` alters >= 5% of Sub-Controller decisions in both `genomic` and `video`.
- **H1_asym (Hierarchical Gain)**: `GM-HRL-v2-Myopic` achieves a statistically significant reduction in weighted deadline violation (`WV`) compared with single-level `GNN-PPO-Myopic` (two-sided paired Wilcoxon test, `p < 0.05` in both domains).

## 3. Fixed Training and Evaluation Configuration
- Domains: `genomic`, `video` (`mw_K=12`, `stat_share=0.2`, `rho=0.8`)
- Budget: `n_train=100`, `iters=400`, `episodes_per_iter=12`, `bc_iters=40`, `critic_warmup=10`, `expert=wedf`, seeds `0, 1, 2`.
