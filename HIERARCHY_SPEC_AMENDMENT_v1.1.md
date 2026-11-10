# Amendment v1.1: load-sensitivity sweep (descriptive)

* **Parent protocol:** `HIERARCHY_SPEC.md` (v1.0), commit `4fec357`
  (2026-11-11 04:31 +0800). That file is not changed by this amendment.
* **Date:** 2026-11-11. The commit time and hash of this file are recorded by
  git and GitHub; they are not written into this file.
* **Script:** `run_hier_load.sh` (a separate script, so that the running
  `run_hier.sh` is not modified).

## 1. Scope

This amendment adds one descriptive evaluation. It does not change anything
in v1.0:
* the checks M1 and M2 and the hypotheses H1 and H2, their criteria and the
  decision script `scripts/decide_hier.py`;
* the main setting (K = 12, STAT share 0.2, deadline factors 1.5 and 2,
  load rho = 0.8) and the secondary settings (K = 30; STAT share 0.1 and 0.4);
* the methods, training budget, model selection and training seeds (3).

## 2. Addition

The test-set evaluation is repeated at offered loads rho = 0.7, 0.9 and 0.95,
in addition to the main-setting load 0.8.

## 3. Procedure

1. **Evaluation only.** The policies are the K = 12 policies trained by
   `run_hier.sh` at rho = 0.8. No policy is trained or fine-tuned at another
   load, as in the robustness test of the main study (Section 6.4 of the
   paper). The script refuses to run before all training runs have finished.
2. **Instances.** The same test batches as the main setting (k = 0..19,
   K = 12, STAT share 0.2) with realisations 1000..1002. Changing rho changes
   only the release times (rescaled to the new offered load); workflows,
   STAT assignments and deadline factors are unchanged, and deadlines are
   recomputed from the rescaled release times (D_w = r_w + f_w * CP_w).
3. **Methods.** All methods of the main comparison, to avoid choosing
   comparators: FIFO, HEFT, EDF, weighted EDF, GA, PPO, GNN-PPO, GM-HRL and
   GM-HRL v2 (learned methods: 3 training seeds each). The M1 measurement
   (share of decisions that change under the neutral goal) is recorded for
   the hierarchical models.
4. **Metrics.** Weighted deadline-violation rate WV, STAT and routine
   violation rates separately, weighted tardiness, makespan.

## 4. Statistical treatment

As for the other secondary settings of v1.0: means, and paired two-sided
Wilcoxon signed-rank p-values against GM-HRL v2, reported descriptively. No
hypothesis, threshold or decision rule applies to this sweep.

## 5. Reporting

The results at all three loads are reported in Section 6.5 of the paper
whatever their direction or size, including when the hierarchy shows no
gain at any load, a gain at some loads, or different effects on STAT and
routine workflows. The paper states that the sweep is descriptive
(evaluation only, no retraining, 3 seeds, no criterion).

## 6. Interpretation

A statement that the hierarchy helps at high load, or that some load is a
boundary of its effectiveness, would require a separately pre-registered
test on fresh data. This sweep is not such a test and is not used to make
such a statement; it may be described only as an observation.

## 7. What was known when this amendment was written

At the time of this commit, the data, the GA rules and the first training
wave (PPO, both domains, seeds 0-2) had started; the only results seen were
the progress lines of the PPO training logs (behaviour-cloning accuracy and
validation WV on the selection batches k = 60..64). No result of GNN-PPO,
GM-HRL or GM-HRL v2 existed, and nothing had been evaluated at any load
other than 0.8.
