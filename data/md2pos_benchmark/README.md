# MD2POS benchmark, version 1.1

Synthetic instances for multi-dimensional dynamic parallel online scheduling (MD2POS) with runtime-revealed dependencies (RRD).

* 4 domains (`genomic`, `video`, `ai_ft`, `financial`) × 5 scales (N = 50, 100, 200, 500, 1000) × 20 instances (`instance_seed0000` … `instance_seed0019`), each stored as `.json` and `.npz`.
* These 400 instances are the **test set** of the paper. Training and validation instances are generated on the fly with the same generator (`gmhrl/generate.py`, k = 100–199 for training and k = 50–54 for validation), so they never coincide with test instances.

## What is synthetic

Stage names and DAG shapes follow four application pipelines (whole-genome sequencing, video transcoding, model fine-tuning, batch risk reporting). Processing times, resource demands, arrival times and branching probabilities are **synthetic parameters chosen by the authors**; they are not calibrated from execution traces. Measured data from a real pipeline are reported separately in the paper (Section 6.3).

## Changes from v1.0

1. **Reproducible seeds.** The v1.0 generator seeded instance k with `k*1000 + hash(domain) % 1000`, and Python randomises string hashes per process. The offsets have been recovered (genomic 851, video 176, ai_ft 496, financial 719); `gmhrl/generate.py` reproduces the stored instances exactly.
2. **Back-edge repair.** In `ai_ft`, 86 of 100 instances contain 1–6 edges (u, v) with u > v, that is, a task that depends on a later-released task; in 16 instances these close a cycle. The files are unchanged; the edges are removed when an instance is loaded (`gmhrl.instance.load_instance(..., repair_back_edges=True)`, the default).

## File format

```python
import json
inst = json.load(open("genomic/N200/instance_seed0000.json"))
inst["metadata"]   # domain, N, software (stage) names, p_rrd, ...
inst["tasks"]      # per task: sw_type, proc_time_s, cpu, ram, gpu, disk, priority, arrival_time_s, rrd_prob
inst["edges"]      # [[src, dst], ...] precedence edges
```

The `.npz` files hold the same data as arrays (`sw_types`, `proc_times`, `cpu_demand`, `ram_demand`, `gpu_demand`, `disk_demand`, `priorities`, `arrival_times`, `rrd_probs`, `edges`).

## Licence

CC BY 4.0.
