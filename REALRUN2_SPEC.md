# Real genomic pipeline: second, independent batch (v2.1)

Pre-registration. This file, `realrun/wgs_chr22/prepare_batch2.sh` and the
two-line changes to `realrun/setup_1000g_chr22.sh` (optional `EXCLUDE` list)
and `realrun/wgs_chr22/run_real.sh` (optional `RUNS` directory; defaults
unchanged) are committed before the new samples are drawn.

## 1. Purpose

Independent replication of the real-pipeline experiment (Section 6.3) on 50
samples that were not part of the first batch or the pilot.

## 2. Samples (data isolation)

* Same collection as batch 1 (1000 Genomes high-coverage, 30x, GRCh38), same
  restriction to chromosome 22, same download and conversion
  (`setup_1000g_chr22.sh`).
* 50 samples drawn at random with seed 2027 after excluding the 50 samples of
  batch 1 and the 5 pilot samples; `prepare_batch2.sh` checks that there is no
  overlap.
* No new pilot: the branch statistic and threshold (duplicate rate > 0.10:
  GATK + FreeBayes, otherwise GATK only), the stage resources and estimated
  durations, the SLA threshold (660 s per task) and every other entry of
  `wgs_chr22.yaml` are unchanged.
* Arrivals: `make_batch.py` with the same mean gap (240 s, offered load 0.8 as
  in batch 1) and the same seed (2026).

## 3. Runs

* Schedulers: FIFO, HEFT and GM-HRL with the checkpoint of batch 1
  (`runs_v2/gmhrl_genomic_N500_s0/best.pt`), no retraining.
* One run per scheduler, in the order HEFT, GM-HRL, FIFO (fixed now; batch 1
  used FIFO-HEFT-GM-HRL and the reverse).
* Same workstation and settings as batch 1. Results go to `realruns2/`.
* Command: `W=<batch-2 data> RUNS=$PWD/realruns2 ORDER="heft gmhrl fifo"
  bash realrun/wgs_chr22/run_real.sh real`, then `... analyze`.

## 4. Pre-registered criterion and reporting

* **P1.** GM-HRL has the lowest task-level SLA violation rate of the three
  schedulers (as in both runs of batch 1).
* Reported without a criterion: sample-level SLA violation rate, makespan
  (GM-HRL relative to HEFT and FIFO), mean sample response time, utilisation,
  gap to the offline CP-SAT bound, decision latency, number of triggered
  branches.
* With one run per scheduler there is no between-run test; the paper reports
  the values of batch 2 next to those of batch 1 and states whether P1 holds.
* Results are reported whatever their outcome. If a run fails for a technical
  reason (crash, disk full), it is repeated with the same settings and the
  failure is reported.
