#!/usr/bin/env python3
"""
Offline clairvoyant benchmark for the production batch.

Once the production run has finished, every runtime-revealed dependency (RRD)
branch is known from the logs. This script builds the corresponding offline
clairvoyant MD2POS instance and reports, in the time unit of the input:

  * an analytical lower bound  LB_ana = max(critical path, resource-volume bound,
    per-type load bound), each measured from the earliest release;
  * a CP-SAT solution under a time limit: best makespan found (incumbent) and
    best proven lower bound (best bound);
  * the gap of every observed schedule (GM-HRL, HEFT, FIFO, ...) to the best
    available lower bound:  gap = (C_obs - LB) / LB.

Makespan is measured as in Eq. (total_makespan): max_i C_i - min_i A_i.

Usage
-----
    python3 clairvoyant_bound.py instance.json --time-limit 3600 --workers 8
    python3 clairvoyant_bound.py instance.json --time-limit 3600 --out result.json

Input format (JSON)
-------------------
{
  "time_unit": "s",
  "capacity":  {"cpu": 32, "ram": 128, "gpu": 4},          # R^limit
  "instances": {"bwa": 8, "samtools": 8, "gatk": 4},        # optional: M_s per processor type
  "tasks": [
    {"id": "S01_bwa", "release": 0,
     "durations": {"bwa": 520},                             # eligible types -> processing time
     "demand": {"cpu": 8, "ram": 5.2, "gpu": 0}},           # resource footprint while running
    ...
  ],
  "edges": [["S01_bwa", "S01_samtools"], ...],              # REALISED precedences (all RRD branches resolved)
  "observed": {"GM-HRL": 9126, "HEFT": 10572, "FIFO": 11010} # optional: makespans of the actual runs
}

Notes
-----
* Use the realised duration of each task. If a task could have run on several
  types but only one duration was observed, either list only the type it ran on
  (conservative: the bound is then valid only for schedules using the same
  types), or scale the observed duration by T_s'/T_s as in the paper's model
  T_{i,s} = Size_i * T_s and list all eligible types.
* Fractional demands (e.g. GB of RAM) are scaled to integers by --resource-scale.
* Durations are rounded up to integers of the time unit (use seconds).
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque


def load(path):
    with open(path, encoding="utf-8") as f:
        inst = json.load(f)
    ids = [t["id"] for t in inst["tasks"]]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate task ids")
    idset = set(ids)
    for u, v in inst["edges"]:
        if u not in idset or v not in idset:
            sys.exit(f"edge ({u},{v}) refers to an unknown task")
    return inst


def topo_order(ids, edges):
    succ = defaultdict(list)
    indeg = {i: 0 for i in ids}
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1
    q = deque([i for i in ids if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(ids):
        sys.exit("precedence graph contains a cycle")
    return order, succ


def analytical_bounds(inst):
    tasks = {t["id"]: t for t in inst["tasks"]}
    ids = list(tasks)
    a0 = min(t.get("release", 0) for t in tasks.values())
    pmin = {i: min(tasks[i]["durations"].values()) for i in ids}

    # 1) release-aware critical path (head + processing)
    order, succ = topo_order(ids, inst["edges"])
    est = {i: tasks[i].get("release", 0) for i in ids}
    for u in order:
        for v in succ[u]:
            est[v] = max(est[v], est[u] + pmin[u])
    lb_cp = max(est[i] + pmin[i] for i in ids) - a0

    # 2) resource-volume bound per dimension
    lb_vol, vol_detail = 0.0, {}
    for r, cap in inst["capacity"].items():
        if cap <= 0:
            continue
        vol = sum(pmin[i] * tasks[i].get("demand", {}).get(r, 0) for i in ids)
        vol_detail[r] = vol / cap
        lb_vol = max(lb_vol, vol / cap)
    # earliest release is the reference point, so the volume bound applies directly

    # 3) per-type load bound for tasks eligible on exactly one type
    lb_type, type_detail = 0.0, {}
    m = inst.get("instances", {})
    if m:
        load_by_type = defaultdict(float)
        for i in ids:
            d = tasks[i]["durations"]
            if len(d) == 1:
                s = next(iter(d))
                load_by_type[s] += d[s]
        for s, ld in load_by_type.items():
            if m.get(s):
                type_detail[s] = ld / m[s]
                lb_type = max(lb_type, ld / m[s])

    lb = max(lb_cp, math.ceil(lb_vol), math.ceil(lb_type))
    return {
        "critical_path": lb_cp,
        "resource_volume": lb_vol,
        "resource_volume_by_dim": vol_detail,
        "type_load": lb_type,
        "type_load_by_type": type_detail,
        "analytical_LB": lb,
    }


def solve_cpsat(inst, time_limit, workers, resource_scale, hint_lb=0):
    from ortools.sat.python import cp_model

    tasks = inst["tasks"]
    a0 = min(t.get("release", 0) for t in tasks)
    horizon = int(sum(max(t["durations"].values()) for t in tasks)
                  + max(t.get("release", 0) for t in tasks) - a0 + 1)

    mdl = cp_model.CpModel()
    start, end = {}, {}
    alt = defaultdict(list)          # type -> [(interval, task)]
    task_ivs = {}                    # task -> [(interval, presence or None, type)]
    for t in tasks:
        i = t["id"]
        r = int(t.get("release", 0) - a0)
        dmax = max(int(math.ceil(d)) for d in t["durations"].values())
        start[i] = mdl.NewIntVar(r, horizon, f"s_{i}")
        end[i] = mdl.NewIntVar(r, horizon, f"e_{i}")
        opts = list(t["durations"].items())
        if len(opts) == 1:
            s, d = opts[0]
            d = int(math.ceil(d))
            iv = mdl.NewIntervalVar(start[i], d, end[i], f"iv_{i}")
            task_ivs[i] = [(iv, None, s)]
            alt[s].append((iv, i))
        else:
            dur = mdl.NewIntVar(min(int(math.ceil(d)) for _, d in opts), dmax, f"d_{i}")
            mdl.NewIntervalVar(start[i], dur, end[i], f"iv_{i}")
            lits = []
            task_ivs[i] = []
            for s, d in opts:
                d = int(math.ceil(d))
                p = mdl.NewBoolVar(f"x_{i}_{s}")
                iv = mdl.NewOptionalIntervalVar(start[i], d, end[i], p, f"iv_{i}_{s}")
                mdl.Add(dur == d).OnlyEnforceIf(p)
                lits.append(p)
                task_ivs[i].append((iv, p, s))
                alt[s].append((iv, i))
            mdl.AddExactlyOne(lits)

    # precedences (realised)
    for u, v in inst["edges"]:
        mdl.Add(start[v] >= end[u])

    # multi-dimensional capacity: one cumulative per resource dimension
    for r, cap in inst["capacity"].items():
        ivs, dem = [], []
        for t in tasks:
            q = t.get("demand", {}).get(r, 0)
            if q <= 0:
                continue
            for iv, _, _ in task_ivs[t["id"]]:
                ivs.append(iv)
                dem.append(int(round(q * resource_scale)))
        if ivs:
            mdl.AddCumulative(ivs, dem, int(math.floor(cap * resource_scale)))

    # processor instances per type: at most M_s tasks of type s at once
    for s, m in inst.get("instances", {}).items():
        if alt.get(s) and m:
            mdl.AddCumulative([iv for iv, _ in alt[s]], [1] * len(alt[s]), int(m))

    cmax = mdl.NewIntVar(int(hint_lb), horizon, "cmax")
    mdl.AddMaxEquality(cmax, list(end.values()))
    mdl.Minimize(cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    t0 = time.time()
    status = solver.Solve(mdl)
    res = {
        "status": solver.StatusName(status),
        "wall_time_s": round(time.time() - t0, 1),
        "time_limit_s": time_limit,
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res["incumbent"] = int(solver.ObjectiveValue())
        res["best_bound"] = int(math.ceil(solver.BestObjectiveBound()))
        res["solver_gap"] = (res["incumbent"] - res["best_bound"]) / max(res["best_bound"], 1)
    else:
        res["incumbent"] = None
        res["best_bound"] = int(math.ceil(solver.BestObjectiveBound())) if status != cp_model.MODEL_INVALID else None
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resource-scale", type=float, default=100.0,
                    help="multiply fractional demands/capacities by this before rounding (default 100)")
    ap.add_argument("--no-cpsat", action="store_true", help="analytical bound only")
    ap.add_argument("--out", help="write the result as JSON")
    args = ap.parse_args()

    inst = load(args.instance)
    n, e = len(inst["tasks"]), len(inst["edges"])
    unit = inst.get("time_unit", "s")
    print(f"instance: {n} tasks, {e} realised precedence edges, unit = {unit}")

    ana = analytical_bounds(inst)
    print(f"analytical LB = {ana['analytical_LB']}  "
          f"(critical path {ana['critical_path']}, resource volume {ana['resource_volume']:.1f}, "
          f"type load {ana['type_load']:.1f})")

    result = {"n_tasks": n, "n_edges": e, "time_unit": unit, "analytical": ana}
    lb = ana["analytical_LB"]
    if not args.no_cpsat:
        cp = solve_cpsat(inst, args.time_limit, args.workers, args.resource_scale, hint_lb=lb)
        result["cpsat"] = cp
        print(f"CP-SAT: status {cp['status']}, incumbent {cp['incumbent']}, best bound {cp['best_bound']}, "
              f"{cp['wall_time_s']} s")
        if cp.get("best_bound") is not None:
            lb = max(lb, cp["best_bound"])
    result["best_LB"] = lb
    print(f"best lower bound = {lb}")

    obs = inst.get("observed", {})
    if obs:
        result["gaps_to_best_LB"] = {}
        print("gap of observed schedules to the best lower bound:")
        for k, v in obs.items():
            g = (v - lb) / lb
            result["gaps_to_best_LB"][k] = g
            line = f"  {k:<10s} {v:>10.1f}  gap {100*g:6.2f}%"
            inc = result.get("cpsat", {}).get("incumbent")
            if inc:
                line += f"   vs CP-SAT incumbent {100*(v-inc)/inc:+6.2f}%"
            if v < lb:
                line += "   <-- WARNING: below the lower bound; check units, release times and the realised edges"
            print(line)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
