"""Analysis of real-run event logs (events.jsonl written by the dispatcher).

For each run: makespan, task- and sample-level SLA violation rates, resource
utilisation (from declared requests), mean waiting time, RRD events, failures,
measured peak CPU/RSS per stage, decision latency; plus the realised instance
(actual durations, revealed edges) for the offline clairvoyant bound.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

RES = ("cpu", "ram", "gpu", "disk")


def read_events(path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_run(path) -> Dict:
    ev = read_events(path)
    hdr = next(e for e in ev if e["ev"] == "header")
    spec = hdr["spec"]
    units = {r: float(spec["model_units"][r]) for r in RES}
    stages = {s["name"]: s for s in spec["stages"]}
    T = defaultdict(dict)
    edges = set()
    branches, fails = [], []
    done = next((e for e in ev if e["ev"] == "done"), {})
    for e in ev:
        if e["ev"] == "visible":
            t = T[e["task"]]
            t.update(task=e["task"], sample=e["sample"], stage=e["stage"], dynamic=e["dynamic"], rel=e["rel"])
            for p in e.get("preds", []):
                edges.add((p, e["task"]))
            for s in e.get("succ", []):
                edges.add((e["task"], s))
        elif e["ev"] == "dispatch":
            T[e["task"]]["start"] = e["t"]
        elif e["ev"] == "end":
            T[e["task"]].update(end=e["t"], exit=e["exit"], peak_cpu_pct=e.get("peak_cpu_pct"),
                                peak_rss_gb=e.get("peak_rss_gb"))
        elif e["ev"] == "branch":
            branches.append(e)
            for d in e.get("new_tasks", []):
                T[d]["parent"] = e["task"]
        elif e["ev"] == "fail":
            fails.append(e)
    return {"header": hdr, "tasks": dict(T), "edges": edges, "branches": branches, "fails": fails,
            "done": done, "units": units, "stages": stages}


def metrics(run: Dict) -> Dict:
    hdr = run["header"]
    spec = hdr["spec"]
    sla = float(spec["sla_seconds"])
    sample_sla = spec.get("sample_sla_seconds")
    cap = np.array(hdr["capacity_units"])
    ok = [t for t in run["tasks"].values() if t.get("exit") == 0]
    rel = np.array([t["rel"] for t in ok]); st = np.array([t["start"] for t in ok]); en = np.array([t["end"] for t in ok])
    t0 = float(min(t["rel"] for t in run["tasks"].values() if "rel" in t))
    mk = float(en.max() - t0)
    dem = np.array([[run["stages"][t["stage"]].get(r, 0.0) / run["units"][r] for r in RES] for t in ok])
    util = ((en - st)[:, None] * dem).sum(0) / (cap * mk)
    by_sample = defaultdict(list)
    for t in ok:
        by_sample[t["sample"]].append(t)
    arrivals = {s: a for s, a in hdr["samples"]}
    samp_resp = np.array([max(x["end"] for x in v) - arrivals[s] for s, v in by_sample.items()])
    per_stage = defaultdict(lambda: {"n": 0, "dur": [], "peak_cpu_pct": [], "peak_rss_gb": []})
    for t in ok:
        d = per_stage[t["stage"]]
        d["n"] += 1
        d["dur"].append(t["end"] - t["start"])
        if t.get("peak_cpu_pct") is not None:
            d["peak_cpu_pct"].append(t["peak_cpu_pct"]); d["peak_rss_gb"].append(t["peak_rss_gb"])
    stage_stats = {k: {"n": v["n"], "duration_mean_s": float(np.mean(v["dur"])),
                       "duration_cv": float(np.std(v["dur"]) / max(np.mean(v["dur"]), 1e-9)),
                       "peak_cpu_pct_max": float(np.max(v["peak_cpu_pct"])) if v["peak_cpu_pct"] else None,
                       "peak_rss_gb_max": float(np.max(v["peak_rss_gb"])) if v["peak_rss_gb"] else None}
                   for k, v in per_stage.items()}
    out = {
        "policy": hdr["policy"], "dry_run": hdr["dry_run"],
        "n_samples": len(hdr["samples"]), "n_tasks_executed": len(ok),
        "n_failed_tasks": len(run["fails"]), "n_rrd_events": sum(1 for b in run["branches"] if b["new_tasks"]),
        "n_branch_points": len(run["branches"]),
        "makespan_s": mk,
        "task_sla_threshold_s": sla,
        "task_sla_violation_rate": float(np.mean((en - rel) > sla)),
        "task_sla_violations": int(np.sum((en - rel) > sla)),
        "sample_response_mean_s": float(samp_resp.mean()),
        "mean_wait_s": float(np.mean(st - rel)),
        **{f"util_{r}": float(util[k]) for k, r in enumerate(RES)},
        "decision_ms_mean": run["done"].get("decision_ms_mean"),
        "decision_ms_p95": run["done"].get("decision_ms_p95"),
        "stage_stats": stage_stats,
    }
    if sample_sla:
        out["sample_sla_threshold_s"] = float(sample_sla)
        out["sample_sla_violation_rate"] = float(np.mean(samp_resp > float(sample_sla)))
    return out


def realised_instance(run: Dict) -> Dict:
    """Executed tasks with actual durations and realised edges, in the input
    format of the clairvoyant bound (releases of RRD-inserted tasks are set to
    their parent's release, as in the simulator)."""
    import math
    hdr = run["header"]
    ok = {i: t for i, t in run["tasks"].items() if t.get("exit") == 0}
    tasks = []
    for i, t in ok.items():
        rel = t["rel"] if not t["dynamic"] else run["tasks"][t["parent"]]["rel"]
        st = run["stages"][t["stage"]]
        tasks.append({"id": f"t{i}", "release": float(rel),
                      "durations": {st["model_type"]: float(max(1, math.ceil(t["end"] - t["start"])))},
                      "demand": {r: float(st.get(r, 0.0)) / run["units"][r] for r in RES}})
    m = int(hdr["spec"].get("instances_per_stage", 4))
    types = sorted({run["stages"][t["stage"]]["model_type"] for t in ok.values()})
    return {"time_unit": "s", "capacity": {r: float(c) for r, c in zip(RES, hdr["capacity_units"])},
            "instances": {ty: m for ty in types}, "tasks": tasks,
            "edges": [[f"t{u}", f"t{v}"] for u, v in sorted(run["edges"]) if u in ok and v in ok]}
