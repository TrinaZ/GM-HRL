#!/usr/bin/env python3
"""Real-pipeline experiment (Section 6.3 of the paper).

Run one batch under one policy:
  python scripts/realrun.py run --spec realrun/example_wgs.yaml --samples realrun/samples.tsv \
      --policy fifo --run_dir realruns/batch1_fifo
  python scripts/realrun.py run ... --policy heft  --run_dir realruns/batch1_heft
  python scripts/realrun.py run ... --policy gmhrl --ckpt runs/gmhrl_genomic_N200_s0/best.pt \
      --run_dir realruns/batch1_gmhrl
Rehearse without the cluster (commands replaced by sleeps, 1 s -> 0.005 s):
  python scripts/realrun.py run ... --dry_run --time_scale 0.005
Analyse and compute the clairvoyant bound on every realised run:
  python scripts/realrun.py analyze realruns/batch1_* --bound --out realruns/summary
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gmhrl.realrun.spec import load_spec  # noqa: E402
from gmhrl.realrun.dispatcher import Dispatcher  # noqa: E402
from gmhrl.realrun import analyze as A  # noqa: E402
from gmhrl import bounds as B  # noqa: E402


def read_samples(path, interval):
    rows = []
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(l for l in f if l.strip() and not l.startswith("#")):
            parts = line.strip().split("\t") if "\t" in line else line.strip().split(",")
            sid = parts[0].strip()
            arr = float(parts[1]) if len(parts) > 1 and parts[1].strip() else k * interval
            rows.append((sid, arr))
    return rows


def make_policy(name, ckpt):
    from gmhrl.baselines import fifo, heft
    if name == "fifo":
        return fifo
    if name == "heft":
        return heft
    if name == "orhg":
        from gmhrl.orhg import make_orhg_policy
        return make_orhg_policy(1.0, 1)
    if name == "gmhrl":
        from gmhrl.agent import GMHRLAgent
        return GMHRLAgent.load(ckpt).online_policy()
    raise ValueError(name)


def cmd_run(a):
    spec = load_spec(a.spec)
    samples = read_samples(a.samples, a.arrival_interval)
    pol = make_policy(a.policy, a.ckpt)
    d = Dispatcher(spec, samples, pol, a.policy, a.run_dir, dry_run=a.dry_run, time_scale=a.time_scale,
                   poll_seconds=a.poll, seed=a.seed, extra_header={"ckpt": a.ckpt})
    path = d.run()
    m = A.metrics(A.parse_run(path))
    json.dump(m, open(Path(a.run_dir) / "metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "stage_stats"}, indent=2))


def cmd_analyze(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rd in a.run_dirs:
        run = A.parse_run(Path(rd) / "events.jsonl")
        m = A.metrics(run)
        m["run_dir"] = rd
        if a.bound:
            inst = A.realised_instance(run)
            json.dump(inst, open(Path(rd) / "realised_instance.json", "w"))
            ana = B.analytical_bounds(inst)
            cp = B.solve_cpsat(inst, a.bound_time, a.bound_workers, 100.0, hint_lb=ana["analytical_LB"])
            lb = max(ana["analytical_LB"], cp.get("best_bound") or 0)
            # makespan of the realised instance is measured from its earliest release
            m.update(analytical_LB_s=ana["analytical_LB"], cpsat_status=cp["status"],
                     cpsat_incumbent_s=cp.get("incumbent"), cpsat_bound_s=cp.get("best_bound"),
                     best_LB_s=lb, gap_to_clairvoyant_LB=(m["makespan_s"] - lb) / lb)
        json.dump(m, open(Path(rd) / "metrics.json", "w"), indent=2)
        rows.append(m)
    keys = [k for k in rows[0] if k != "stage_stats"]
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{'run':40s} {'policy':7s} {'makespan':>9s} {'taskSLA':>8s} {'CPU':>6s}" + ("   gap" if a.bound else ""))
    for m in rows:
        line = (f"{Path(m['run_dir']).name:40s} {m['policy']:7s} {m['makespan_s']:9.0f} "
                f"{m['task_sla_violation_rate']:8.3f} {m['util_cpu']:6.3f}")
        if a.bound:
            line += f" {100 * m['gap_to_clairvoyant_LB']:5.1f}%"
        print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--samples", required=True, help="TSV/CSV: sample_id[, arrival offset in s]")
    r.add_argument("--arrival_interval", type=float, default=0.0,
                   help="arrival spacing (s) when the samples file has no arrival column")
    r.add_argument("--policy", required=True, choices=["fifo", "heft", "orhg", "gmhrl"])
    r.add_argument("--ckpt", default=None)
    r.add_argument("--run_dir", required=True)
    r.add_argument("--dry_run", action="store_true")
    r.add_argument("--time_scale", type=float, default=1.0)
    r.add_argument("--poll", type=float, default=1.0, help="polling interval (pipeline seconds)")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("analyze")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--bound", action="store_true")
    s.add_argument("--bound_time", type=float, default=3600)
    s.add_argument("--bound_workers", type=int, default=8)
    s.add_argument("--out", default="realruns/summary")
    s.set_defaults(fn=cmd_analyze)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
