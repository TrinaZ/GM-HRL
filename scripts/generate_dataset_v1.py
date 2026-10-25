"""MD2POS benchmark v1.0 generator (restored from the commented-out block of DDA-HRL/generate_dataset.py).

Usage:
  python generate_dataset.py                  # generate all
  python generate_dataset.py --domain genomic --N 200 --seeds 10
  python generate_dataset.py --export_json    # also write JSON instances

Output:
  ./md2pos_benchmark/
  ├── README.md
  ├── metadata.json
  ├── genomic/   N50/ N100/ N200/ N500/ N1000/
  ├── video/     ...
  ├── ai_ft/     ...
  └── financial/ ...

  Each *.npz instance file contains:
    tasks     : structured array with fields
                  id, sw_type, proc_time, cpu, ram, gpu, disk, priority
    edges     : [E, 2] int32  (src, dst) dependency edges
    rrd_probs : [T] float32   per-task RRD branching probability
    metadata  : dict (domain, N, seed, scale_N, ...)
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Domain Configurations
#  All parameters are grounded in published literature.
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_CONFIGS: Dict[str, Dict] = {

    "genomic": {
        "description": (
            "Whole-genome sequencing (WGS) bioinformatics pipeline. "
            "Stages: alignment (BWA-MEM) → sort/index (SAMtools) → "
            "variant calling (FreeBayes or GATK HaplotypeCaller). "
            "Dynamic branching (RRD) models stochastic heterozygosity: "
            "samples triggering the dual-caller path with probability p_rrd."
        ),
        "citation": (
            "Stage structure modelled on a whole-genome sequencing pipeline (cf. 1000 Genomes Project Consortium, 2015). Durations, demands and branching probabilities are synthetic parameters set by the authors; they are not calibrated from execution traces."
        ),
        # Software types (processing stages)
        "software_types": [
            "BWA-MEM",       # 0: read alignment (CPU-heavy)
            "SAMtools-sort", # 1: sort BAM
            "SAMtools-idx",  # 2: index BAM
            "FreeBayes",     # 3: variant calling path A
            "GATK-HC",       # 4: variant calling path B (RRD branch)
            "bcftools",      # 5: VCF merge/filter
            "FastQC",        # 6: quality control
            "MultiQC",       # 7: report aggregation
        ],
        # Processing time: LogNormal(mu, sigma) in seconds
        "proc_time": {
            "mu":    [5.2, 3.1, 1.8, 4.8, 5.5, 2.2, 2.5, 3.0],
            "sigma": [1.5, 0.9, 0.6, 1.4, 1.6, 0.8, 0.7, 0.9],
        },
        # Resource demand: fraction of node capacity
        "cpu":  [0.70, 0.30, 0.10, 0.60, 0.65, 0.20, 0.25, 0.15],
        "ram":  [0.50, 0.40, 0.20, 0.45, 0.50, 0.25, 0.20, 0.30],
        "gpu":  [0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
        "disk": [0.60, 0.50, 0.30, 0.35, 0.40, 0.25, 0.20, 0.15],
        # DAG structure: pipeline with fan-out at variant calling
        "dag_type": "pipeline_fanout",
        "p_rrd":    0.30,   # probability of triggering dual-caller branch
        "avg_degree": 2.8,
        "priority_dist": "uniform",
    },

    "video": {
        "description": (
            "Video transcoding and streaming preparation pipeline. "
            "Stages: decode → scene-detect → multi-resolution encode "
            "(H.264/H.265, 4K/1080p/720p/480p) → package (HLS/DASH) → "
            "thumbnail generation → upload. "
            "RRD branching models adaptive bitrate decisions triggered "
            "by per-scene complexity analysis."
        ),
        "citation": (
            "Stage structure modelled on a video transcoding pipeline. Durations, demands and branching probabilities are synthetic parameters set by the authors; they are not calibrated from execution traces."
        ),
        "software_types": [
            "Decode",        # 0: decode input video
            "SceneDetect",   # 1: scene complexity analysis
            "Encode-4K",     # 2: 4K H.265 encode (GPU)
            "Encode-1080p",  # 3: 1080p H.264 encode
            "Encode-720p",   # 4: 720p H.264 encode
            "Package-HLS",   # 5: HLS packaging (RRD branch A)
            "Package-DASH",  # 6: DASH packaging (RRD branch B)
            "Thumbnail",     # 7: thumbnail extraction
        ],
        "proc_time": {
            "mu":    [3.2, 1.5, 5.8, 4.2, 3.5, 2.0, 2.1, 1.2],
            "sigma": [0.8, 0.5, 1.8, 1.2, 1.0, 0.6, 0.6, 0.4],
        },
        "cpu":  [0.40, 0.25, 0.30, 0.80, 0.70, 0.35, 0.35, 0.20],
        "ram":  [0.35, 0.20, 0.40, 0.55, 0.45, 0.30, 0.30, 0.15],
        "gpu":  [0.10, 0.00, 0.85, 0.10, 0.05, 0.00, 0.00, 0.00],
        "disk": [0.70, 0.20, 0.60, 0.55, 0.50, 0.40, 0.40, 0.15],
        "dag_type": "pipeline_fanout",
        "p_rrd":    0.35,
        "avg_degree": 2.4,
        "priority_dist": "deadline_weighted",
    },

    "ai_ft": {
        "description": (
            "Large language model fine-tuning pipeline for enterprise AI services. "
            "Stages: data ingestion → preprocessing/tokenization → "
            "model fine-tuning (GPU-heavy) → evaluation → "
            "checkpoint management → deployment packaging. "
            "RRD branching models conditional evaluation routing: "
            "models exceeding quality threshold proceed to A/B testing."
        ),
        "citation": (
            "Stage structure modelled on a model fine-tuning pipeline (cf. Hu et al., 2022, LoRA). Durations, demands and branching probabilities are synthetic parameters set by the authors; they are not calibrated from execution traces."
        ),
        "software_types": [
            "DataIngest",    # 0: data download/validation
            "Preprocess",    # 1: tokenization, formatting
            "FineTune-GPU",  # 2: GPU training (LoRA/full fine-tune)
            "Evaluate",      # 3: evaluation on held-out set
            "Checkpoint",    # 4: checkpoint management
            "ABTest",        # 5: A/B test routing (RRD branch)
            "Package",       # 6: model packaging for deployment
            "Monitor",       # 7: deployment health monitoring
        ],
        "proc_time": {
            "mu":    [2.5, 3.8, 7.2, 4.5, 1.8, 2.0, 2.2, 1.5],
            "sigma": [0.7, 1.1, 2.2, 1.3, 0.5, 0.6, 0.7, 0.5],
        },
        "cpu":  [0.30, 0.60, 0.20, 0.40, 0.15, 0.30, 0.25, 0.15],
        "ram":  [0.40, 0.65, 0.70, 0.55, 0.30, 0.35, 0.30, 0.20],
        "gpu":  [0.00, 0.10, 0.95, 0.80, 0.10, 0.05, 0.00, 0.00],
        "disk": [0.50, 0.40, 0.60, 0.30, 0.70, 0.20, 0.45, 0.10],
        "dag_type": "multi_stage",
        "p_rrd":    0.45,
        "avg_degree": 3.1,
        "priority_dist": "priority_queue",
    },

    "financial": {
        "description": (
            "Batch risk calculation and regulatory reporting pipeline "
            "for digital financial services. "
            "Stages: market data ingestion → position loading → "
            "risk factor computation (VaR/CVaR) → stress testing → "
            "regulatory aggregation → report generation. "
            "RRD branching models threshold-triggered stress scenario "
            "expansion: high-volatility regimes activate additional "
            "Monte Carlo simulation paths."
        ),
        "citation": (
            "Stage structure modelled on a batch risk-reporting pipeline. Durations, demands and branching probabilities are synthetic parameters set by the authors; they are not calibrated from execution traces."
        ),
        "software_types": [
            "MarketData",    # 0: real-time/EOD market data pull
            "PositionLoad",  # 1: portfolio position loading
            "RiskFactors",   # 2: risk factor computation (CPU-heavy)
            "VaR-MC",        # 3: Monte Carlo VaR simulation
            "StressTest",    # 4: stress scenario testing (RRD branch)
            "Aggregation",   # 5: cross-desk aggregation
            "Reporting",     # 6: regulatory report generation
            "Audit",         # 7: audit trail logging
        ],
        "proc_time": {
            "mu":    [1.8, 2.5, 4.2, 6.5, 5.8, 3.2, 2.8, 1.5],
            "sigma": [0.5, 0.7, 1.2, 2.0, 1.8, 0.9, 0.8, 0.4],
        },
        "cpu":  [0.20, 0.35, 0.85, 0.90, 0.88, 0.50, 0.30, 0.10],
        "ram":  [0.25, 0.45, 0.60, 0.75, 0.70, 0.55, 0.35, 0.15],
        "gpu":  [0.00, 0.00, 0.05, 0.30, 0.25, 0.00, 0.00, 0.00],
        "disk": [0.30, 0.40, 0.20, 0.25, 0.30, 0.35, 0.60, 0.50],
        "dag_type": "pipeline_parallel",
        "p_rrd":    0.30,
        "avg_degree": 2.6,
        "priority_dist": "deadline_weighted",
    },
}

SCALES = [50, 100, 200, 500, 1000]

# ─────────────────────────────────────────────────────────────────────────────
#  DAG topology generators
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline_fanout(N: int, num_sw: int, rng: np.random.Generator,
                            p_rrd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline with fan-out at branching stages.
    Models genomic & video: linear stages with RRD-induced parallel branches.
    Returns: (sw_types [N], edges [E, 2])
    """
    sw_types = np.zeros(N, dtype=np.int32)
    edges: List[Tuple[int, int]] = []
    stage_nodes: List[List[int]] = [[] for _ in range(num_sw)]

    # Assign tasks to stages in proportion to domain complexity
    base = N // num_sw
    remainder = N % num_sw
    counts = [base + (1 if i < remainder else 0) for i in range(num_sw)]

    idx = 0
    for sw, cnt in enumerate(counts):
        for _ in range(cnt):
            sw_types[idx] = sw
            stage_nodes[sw].append(idx)
            idx += 1

    # Add within-stage parallel edges
    for sw in range(1, num_sw):
        prev = stage_nodes[sw - 1]
        curr = stage_nodes[sw]
        if not prev or not curr:
            continue
        # Fan-in: each current node depends on a subset of prev nodes
        for c in curr:
            n_deps = min(len(prev), max(1, int(rng.poisson(1.5))))
            deps = rng.choice(prev, size=n_deps, replace=False)
            for d in deps:
                edges.append((int(d), int(c)))

    # Add RRD branching edges (cross-stage, bypass dependencies)
    n_rrd = int(N * p_rrd)
    for _ in range(n_rrd):
        if len(edges) == 0:
            continue
        src_sw = int(rng.integers(0, num_sw - 2))
        dst_sw = int(rng.integers(src_sw + 1, num_sw))
        if stage_nodes[src_sw] and stage_nodes[dst_sw]:
            s = int(rng.choice(stage_nodes[src_sw]))
            d = int(rng.choice(stage_nodes[dst_sw]))
            if s != d:
                edges.append((s, d))

    edges_arr = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    return sw_types, edges_arr


def _build_multi_stage(N: int, num_sw: int, rng: np.random.Generator,
                        p_rrd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-stage DAG with complex inter-stage dependencies.
    Models AI fine-tuning: data flows between stages with conditional skips.
    """
    sw_types = np.zeros(N, dtype=np.int32)
    edges: List[Tuple[int, int]] = []

    # Random sw assignment weighted by typical stage volumes
    weights = np.ones(num_sw) / num_sw
    # Data prep and training stages tend to have more tasks
    weights[1] = 0.20   # preprocess
    weights[2] = 0.25   # finetune
    weights /= weights.sum()
    sw_types = rng.choice(num_sw, size=N, p=weights).astype(np.int32)

    # Build edges: nodes with earlier sw type feed into later ones
    for i in range(N):
        for j in range(i + 1, N):
            if sw_types[j] > sw_types[i]:
                if rng.random() < 0.15:  # sparse connectivity
                    edges.append((i, j))
                    break  # limit fan-out

    # RRD edges: conditional evaluation triggers
    n_rrd = int(N * p_rrd)
    for _ in range(n_rrd):
        i = int(rng.integers(0, N))
        j = int(rng.integers(i + 1, N + 1)) if i < N - 1 else i
        if i < j and i != j:
            edges.append((i, j % N))

    edges_arr = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    return sw_types, edges_arr


def _build_pipeline_parallel(N: int, num_sw: int, rng: np.random.Generator,
                              p_rrd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline with parallel tracks.
    Models financial: independent desk calculations merge at aggregation.
    """
    sw_types = np.zeros(N, dtype=np.int32)
    edges: List[Tuple[int, int]] = []
    n_tracks = max(2, N // 20)  # parallel processing tracks

    # Distribute tasks across tracks
    track_assign = rng.integers(0, n_tracks, size=N)
    # Stage within track = determines sw_type
    stage_within = rng.integers(0, num_sw - 1, size=N)  # leave last sw for aggregation
    sw_types = stage_within.astype(np.int32)
    # Final aggregation nodes
    n_agg = max(1, N // 10)
    sw_types[-n_agg:] = num_sw - 1  # aggregation stage

    # Within-track sequential edges
    for track in range(n_tracks):
        track_nodes = sorted(np.where(track_assign == track)[0])
        for k in range(len(track_nodes) - 1):
            edges.append((int(track_nodes[k]), int(track_nodes[k + 1])))

    # All tracks merge at aggregation nodes
    for i in range(N - n_agg):
        for j in range(N - n_agg, N):
            if rng.random() < 0.3:
                edges.append((i, j))

    # RRD: stress scenario expansion
    n_rrd = int(N * p_rrd)
    for _ in range(n_rrd):
        src = int(rng.integers(0, N - n_agg))
        dst = int(rng.integers(N - n_agg, N))
        edges.append((src, dst))

    edges_arr = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    return sw_types, edges_arr


DAG_BUILDERS = {
    "pipeline_fanout":    _build_pipeline_fanout,
    "multi_stage":        _build_multi_stage,
    "pipeline_parallel":  _build_pipeline_parallel,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Instance generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_instance(domain: str, N: int, seed: int) -> Dict:
    """
    Generate one MD2POS scheduling problem instance.

    Returns a dict with all fields needed to reconstruct the problem.
    """
    cfg = DOMAIN_CONFIGS[domain]
    num_sw = len(cfg["software_types"])
    rng = np.random.default_rng(seed)

    # Build DAG topology
    dag_builder = DAG_BUILDERS[cfg["dag_type"]]
    sw_types, edges = dag_builder(N, num_sw, rng, cfg["p_rrd"])

    # Remove duplicate and self edges
    if len(edges) > 0:
        edges = np.unique(edges, axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]

    # Sample processing times (LogNormal)
    proc_times = np.zeros(N, dtype=np.float32)
    mu_arr    = np.array(cfg["proc_time"]["mu"])
    sig_arr   = np.array(cfg["proc_time"]["sigma"])
    for i, sw in enumerate(sw_types):
        # Convert mean/std of the lognormal random variable to μ,σ of log
        m, s = mu_arr[sw], sig_arr[sw]
        # Direct lognormal sample (in seconds, scale to ~60-600s range)
        raw = rng.lognormal(np.log(m), s / m)
        proc_times[i] = float(np.clip(raw * 60, 30, 3600))

    # Per-task resource demands indexed by sw_type
    cpu_demand  = np.zeros(N, dtype=np.float32)
    ram_demand  = np.zeros(N, dtype=np.float32)
    gpu_demand  = np.zeros(N, dtype=np.float32)
    disk_demand = np.zeros(N, dtype=np.float32)
    for i, sw in enumerate(sw_types):
        cpu_demand[i]  = float(np.clip(rng.normal(cfg["cpu"][sw],  0.05), 0.05, 1.0))
        ram_demand[i]  = float(np.clip(rng.normal(cfg["ram"][sw],  0.05), 0.05, 1.0))
        gpu_demand[i]  = float(np.clip(rng.normal(cfg["gpu"][sw],  0.02), 0.00, 1.0))
        disk_demand[i] = float(np.clip(rng.normal(cfg["disk"][sw], 0.05), 0.05, 1.0))

    # Priority assignment
    if cfg["priority_dist"] == "uniform":
        priorities = rng.uniform(0.1, 1.0, N).astype(np.float32)
    elif cfg["priority_dist"] == "deadline_weighted":
        # Higher priority for tasks on critical path (approximated by degree)
        if len(edges) > 0:
            out_deg = np.bincount(edges[:, 0], minlength=N)
            priorities = (out_deg / (out_deg.max() + 1e-6) * 0.8 + 0.2).astype(np.float32)
        else:
            priorities = rng.uniform(0.2, 1.0, N).astype(np.float32)
    else:  # priority_queue
        priorities = rng.exponential(0.5, N).clip(0.1, 1.0).astype(np.float32)

    # Task arrival times (Poisson process)
    inter_arrival = rng.exponential(scale=30.0 / N * 50, size=N)  # scale with N
    arrival_times = np.cumsum(inter_arrival).astype(np.float32)

    # RRD branch probability per task
    rrd_probs = np.where(
        rng.random(N) < cfg["p_rrd"],
        rng.uniform(0.2, 0.6, N),
        0.0
    ).astype(np.float32)

    return {
        "domain":       domain,
        "N":            N,
        "seed":         seed,
        "num_sw":       num_sw,
        "sw_types":     sw_types,           # [N] int32
        "proc_times":   proc_times,         # [N] float32  seconds
        "cpu_demand":   cpu_demand,         # [N] float32  fraction
        "ram_demand":   ram_demand,         # [N] float32  fraction
        "gpu_demand":   gpu_demand,         # [N] float32  fraction
        "disk_demand":  disk_demand,        # [N] float32  fraction
        "priorities":   priorities,         # [N] float32  [0,1]
        "arrival_times": arrival_times,     # [N] float32  seconds
        "rrd_probs":    rrd_probs,          # [N] float32  [0,1]
        "edges":        edges,              # [E, 2] int32  (src, dst)
        "software_names": cfg["software_types"],
        "dag_type":     cfg["dag_type"],
        "p_rrd":        cfg["p_rrd"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset generation & statistics
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(domains: List[str], scales: List[int],
                     n_seeds: int, output_dir: str,
                     export_json: bool = False):
    """Generate the full benchmark dataset."""
    out_root = Path(output_dir)
    stats_all = {}

    print("\n" + "=" * 70)
    print("  MD2POS Benchmark Dataset Generator")
    print("=" * 70)

    for domain in domains:
        cfg = DOMAIN_CONFIGS[domain]
        domain_dir = out_root / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        stats_all[domain] = {}

        print(f"\n  Domain: {domain.upper()}")
        print(f"  Stages: {cfg['software_types']}")

        for N in scales:
            scale_dir = domain_dir / f"N{N}"
            scale_dir.mkdir(parents=True, exist_ok=True)

            proc_t_all, n_edges_all, rrd_all = [], [], []

            for seed in range(n_seeds):
                inst = generate_instance(domain, N, seed=seed * 1000 + hash(domain) % 1000)

                # Save as .npz
                save_path = scale_dir / f"instance_seed{seed:04d}.npz"
                np.savez_compressed(
                    save_path,
                    sw_types    = inst["sw_types"],
                    proc_times  = inst["proc_times"],
                    cpu_demand  = inst["cpu_demand"],
                    ram_demand  = inst["ram_demand"],
                    gpu_demand  = inst["gpu_demand"],
                    disk_demand = inst["disk_demand"],
                    priorities  = inst["priorities"],
                    arrival_times = inst["arrival_times"],
                    rrd_probs   = inst["rrd_probs"],
                    edges       = inst["edges"],
                )

                # Optionally export JSON (human-readable, larger)
                if export_json:
                    json_path = scale_dir / f"instance_seed{seed:04d}.json"
                    json_inst = {
                        "metadata": {
                            "domain": domain,
                            "N": N,
                            "seed": seed,
                            "num_sw": inst["num_sw"],
                            "software_names": inst["software_names"],
                            "dag_type": inst["dag_type"],
                            "p_rrd": inst["p_rrd"],
                            "n_edges": len(inst["edges"]),
                        },
                        "tasks": [
                            {
                                "id": i,
                                "sw_type": int(inst["sw_types"][i]),
                                "sw_name": inst["software_names"][int(inst["sw_types"][i])],
                                "proc_time_s": round(float(inst["proc_times"][i]), 2),
                                "cpu": round(float(inst["cpu_demand"][i]), 3),
                                "ram": round(float(inst["ram_demand"][i]), 3),
                                "gpu": round(float(inst["gpu_demand"][i]), 3),
                                "disk": round(float(inst["disk_demand"][i]), 3),
                                "priority": round(float(inst["priorities"][i]), 3),
                                "arrival_time_s": round(float(inst["arrival_times"][i]), 2),
                                "rrd_prob": round(float(inst["rrd_probs"][i]), 3),
                            }
                            for i in range(N)
                        ],
                        "edges": inst["edges"].tolist(),
                    }
                    with open(json_path, "w") as f:
                        json.dump(json_inst, f, indent=2)

                # Collect stats
                proc_t_all.extend(inst["proc_times"].tolist())
                n_edges_all.append(len(inst["edges"]))
                rrd_all.append((inst["rrd_probs"] > 0).mean())

            # Per-scale statistics
            stats = {
                "n_instances": n_seeds,
                "avg_proc_time_s": float(np.mean(proc_t_all)),
                "std_proc_time_s": float(np.std(proc_t_all)),
                "avg_n_edges": float(np.mean(n_edges_all)),
                "avg_rrd_fraction": float(np.mean(rrd_all)),
            }
            stats_all[domain][f"N{N}"] = stats

            print(f"    N={N:5d}: {n_seeds} instances | "
                  f"edges={stats['avg_n_edges']:.0f} | "
                  f"proc={stats['avg_proc_time_s']:.0f}s | "
                  f"rrd={stats['avg_rrd_fraction']:.2%}")

    # Write metadata.json
    metadata = {
        "dataset": "MD2POS Benchmark v1.0",
        "version": "1.0.0",
        "created": "2026-03",
        "description": (
            "Multidimensional Dynamic Parallel Online Scheduling (MD2POS) benchmark. "
            "Four application domains × five instance scales. "
            "Generated for reproducible research in hierarchical reinforcement "
            "learning for scheduling. See README.md for details."
        ),
        "paper": (
            "A Graph-Mediated Hierarchical Reinforcement Learning Approach "
            "for the MD2POS Problem in Digital Service Supply Chains. "
            "Submitted to Computers & Operations Research, 2026."
        ),
        "domains": list(domains),
        "scales": scales,
        "n_seeds_per_scale": n_seeds,
        "total_instances": len(domains) * len(scales) * n_seeds,
        "domain_descriptions": {
            d: DOMAIN_CONFIGS[d]["description"] for d in domains
        },
        "domain_citations": {
            d: DOMAIN_CONFIGS[d]["citation"] for d in domains
        },
        "statistics": stats_all,
        "file_format": {
            "type": "numpy .npz compressed",
            "fields": {
                "sw_types":     "[N] int32   — software type index [0, num_sw)",
                "proc_times":   "[N] float32 — processing time in seconds",
                "cpu_demand":   "[N] float32 — CPU demand as fraction of node capacity",
                "ram_demand":   "[N] float32 — RAM demand as fraction of node capacity",
                "gpu_demand":   "[N] float32 — GPU demand as fraction of node capacity",
                "disk_demand":  "[N] float32 — disk demand as fraction of node capacity",
                "priorities":   "[N] float32 — task priority in [0, 1]",
                "arrival_times": "[N] float32 — task arrival time in seconds",
                "rrd_probs":    "[N] float32 — RRD branching probability per task",
                "edges":        "[E, 2] int32 — dependency edges (src, dst), 0-based",
            }
        },
        "license": "CC BY 4.0 — free to use with attribution",
        "contact": "See paper for author contact information.",
    }

    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Write README.md
    _write_readme(out_root, metadata, stats_all, domains, scales, n_seeds)

    print(f"\n  ✅ Done! Total instances: {metadata['total_instances']}")
    print(f"     Output: {out_root.resolve()}")
    print(f"     Metadata: {(out_root / 'metadata.json').resolve()}")
    return metadata, stats_all


def _write_readme(out_root: Path, metadata: Dict, stats: Dict,
                  domains: List[str], scales: List[int], n_seeds: int):
    lines = [
        "# MD2POS Benchmark Dataset v1.0",
        "",
        "> **Citation**: If you use this dataset, please cite:",
        "> ",
        f"> {metadata['paper']}",
        "",
        "## Overview",
        "",
        metadata["description"],
        "",
        f"- **Domains**: {', '.join(domains)}",
        f"- **Scales (N)**: {', '.join(map(str, scales))}",
        f"- **Instances per scale**: {n_seeds}",
        f"- **Total instances**: {metadata['total_instances']}",
        f"- **License**: {metadata['license']}",
        "",
        "## Domain Descriptions",
        "",
    ]
    for domain in domains:
        cfg = DOMAIN_CONFIGS[domain]
        lines += [
            f"### {domain.upper()}",
            "",
            cfg["description"],
            "",
            f"**Processing stages**: {', '.join(cfg['software_types'])}",
            "",
            f"**RRD branching probability**: {cfg['p_rrd']}",
            "",
            f"**Citation**: {cfg['citation']}",
            "",
        ]

    lines += [
        "## File Format",
        "",
        "Each instance is stored as a NumPy `.npz` file:",
        "",
        "```python",
        "import numpy as np",
        "inst = np.load('genomic/N200/instance_seed0000.npz')",
        "sw_types   = inst['sw_types']    # [N] int32  — which stage each task belongs to",
        "proc_times = inst['proc_times']  # [N] float32 — processing time (seconds)",
        "edges      = inst['edges']       # [E, 2] int32 — (src, dst) dependency pairs",
        "```",
        "",
        "## Statistics",
        "",
        "| Domain | N | Instances | Avg edges | Avg proc (s) | RRD rate |",
        "|--------|---|-----------|-----------|--------------|----------|",
    ]
    for domain in domains:
        for N in scales:
            key = f"N{N}"
            if key in stats.get(domain, {}):
                s = stats[domain][key]
                lines.append(
                    f"| {domain} | {N} | {s['n_instances']} | "
                    f"{s['avg_n_edges']:.0f} | "
                    f"{s['avg_proc_time_s']:.0f} | "
                    f"{s['avg_rrd_fraction']:.1%} |"
                )

    lines += [
        "",
        "## Usage with DDA-HRL",
        "",
        "```python",
        "from generate_dataset import generate_instance, DOMAIN_CONFIGS",
        "inst = generate_instance('genomic', N=200, seed=42)",
        "# inst['edges'] → dependency graph",
        "# inst['proc_times'] → task durations",
        "# inst['rrd_probs'] → dynamic branching probabilities",
        "```",
        "",
        "Note: all instances are synthetic. Stage names follow the four application domains; timing and resource parameters are set by the authors and are not calibrated from real traces.",
        "",
    ]

    with open(out_root / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics & visualization
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(stats: Dict, domains: List[str], scales: List[int]):
    """Print a formatted comparison table similar to paper Table 7."""
    print("\n" + "=" * 90)
    print("  MD2POS Benchmark — Instance Statistics")
    print("=" * 90)
    header = f"  {'Domain':<12} {'N':>6} {'#Inst':>6} {'AvgEdges':>10} {'AvgProc(s)':>12} {'RRD%':>8}"
    print(header)
    print("  " + "-" * 58)
    for domain in domains:
        for i, N in enumerate(scales):
            key = f"N{N}"
            if key in stats.get(domain, {}):
                s = stats[domain][key]
                d_label = domain if i == 0 else ""
                print(f"  {d_label:<12} {N:>6} {s['n_instances']:>6} "
                      f"{s['avg_n_edges']:>10.1f} "
                      f"{s['avg_proc_time_s']:>12.1f} "
                      f"{s['avg_rrd_fraction']:>8.1%}")
        print("  " + "-" * 58)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate MD2POS benchmark dataset for reproducible research.")
    p.add_argument("--domains", nargs="+",
                   default=["genomic", "video", "ai_ft", "financial"],
                   choices=["genomic", "video", "ai_ft", "financial"])
    p.add_argument("--scales", nargs="+", type=int,
                   default=[50, 100, 200, 500, 1000])
    p.add_argument("--n_seeds", type=int, default=20,
                   help="Number of instances per domain×scale (default: 20)")
    p.add_argument("--output_dir", default="./md2pos_benchmark")
    p.add_argument("--export_json", action="store_true",
                   help="Also export human-readable JSON (much larger files)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 3 seeds, N=[50,100,200] only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick:
        scales = [50, 100, 200]
        n_seeds = 3
    else:
        scales = args.scales
        n_seeds = args.n_seeds

    metadata, stats = generate_dataset(
        domains=args.domains,
        scales=scales,
        n_seeds=n_seeds,
        output_dir=args.output_dir,
        export_json=args.export_json,
    )
    print_comparison_table(stats, args.domains, scales)
