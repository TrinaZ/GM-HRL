"""Dispatcher for real pipeline runs.

Launches the user's stage scripts in the order chosen by a policy (FIFO, HEFT,
ORHG or a trained GM-HRL checkpoint) under the declared resource capacity and
per-stage instance limits, and writes a complete event log (events.jsonl):

  header   spec, policy, samples, capacity, start time
  visible  task becomes known (sample arrival or RRD reveal)
  dispatch task launched (time, command)
  end      task finished (time, exit code, measured peak CPU % / RSS if psutil is available)
  branch   branch decision (metric, revealed stages, new task ids)
  fail     non-zero exit; the sample's remaining tasks are cancelled
  done     run finished

Dry-run mode replaces every command with ``sleep`` of a synthetic duration
(estimate x LogNormal noise x time_scale) and draws branch outcomes from the
stage's rrd_prob, so the whole procedure can be rehearsed without the cluster.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..sim import WAIT
from .live import LiveState
from .spec import PipelineSpec, evaluate_rules

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class Dispatcher:
    def __init__(self, spec: PipelineSpec, samples: List[Tuple[str, float]], policy: Callable,
                 policy_name: str, run_dir: str, dry_run: bool = False, time_scale: float = 1.0,
                 poll_seconds: float = 1.0, seed: int = 0, extra_header: Optional[Dict] = None):
        self.spec, self.policy, self.policy_name = spec, policy, policy_name
        self.state = LiveState(spec, samples)
        self.run_dir = Path(run_dir)
        (self.run_dir / "tasks").mkdir(parents=True, exist_ok=True)
        self.dry_run, self.time_scale, self.poll = dry_run, float(time_scale), float(poll_seconds)
        self.seed = int(seed)
        self._stage_idx = {n: k for k, n in enumerate(spec.stages)}
        self.procs: Dict[int, subprocess.Popen] = {}
        self.usage: Dict[int, Dict] = {}
        self.log = open(self.run_dir / "events.jsonl", "w", encoding="utf-8")
        self.extra_header = extra_header or {}
        self.n_decisions = 0
        self.decision_ms: List[float] = []

    # ------------------------------------------------------------------ utils
    def _emit(self, ev: str, **kw):
        rec = {"ev": ev, "t": round(self.state.t, 3), "wall": time.time(), **kw}
        self.log.write(json.dumps(rec, default=float) + "\n")
        self.log.flush()

    def _now(self) -> float:
        return (time.monotonic() - self._t_start) / (self.time_scale if self.dry_run else 1.0)

    def _info(self, i: int) -> Dict:
        return self.state.task_info[i]

    def _fmt(self, template: str, i: int) -> str:
        inf = self._info(i)
        st = self.spec.stages[inf["stage"]]
        return template.format(sample=inf["sample"], stage=inf["stage"], task=i,
                               workdir=self.spec.workdir, threads=int(max(1, st.request.get("cpu", 1))),
                               mem_gb=int(max(1, st.request.get("ram", 1))))

    # ---------------------------------------------------------------- actions
    def _launch(self, i: int):
        inf = self._info(i)
        st = self.spec.stages[inf["stage"]]
        if self.dry_run:
            rng = np.random.default_rng([self.seed, inf["sample_idx"], self._stage_idx[st.name], 1])
            dur = st.est_seconds * float(rng.lognormal(-0.045, 0.3)) * self.time_scale
            cmd = f"sleep {dur:.4f}"
        else:
            cmd = self._fmt(st.cmd, i)
        out = open(self.run_dir / "tasks" / f"{i:05d}_{inf['sample']}_{inf['stage']}.log", "w")
        env = {**os.environ, "GMHRL_TASK": str(i), "GMHRL_SAMPLE": inf["sample"], "GMHRL_STAGE": inf["stage"]}
        Path(self.spec.workdir).mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT, env=env,
                             cwd=self.spec.workdir, start_new_session=True)
        self.state.start_task(i, self.state.t)
        self.procs[i] = p
        self.usage[i] = {"peak_cpu_pct": 0.0, "peak_rss_gb": 0.0, "procs": {}}
        self._emit("dispatch", task=i, sample=inf["sample"], stage=inf["stage"], cmd=cmd,
                   free_after=self.state.free.tolist())

    def _branch(self, i: int) -> Tuple[Optional[float], List[str]]:
        st = self.spec.stages[self._info(i)["stage"]]
        b = st.branch
        rules = b.get("rules", [])
        if self.dry_run:
            inf = self._info(i)
            rng = np.random.default_rng([self.seed, inf["sample_idx"], self._stage_idx[st.name], 2])
            alt = rng.random() < st.rrd_prob
            first = next((r.get("reveal", []) for r in rules if "if" in r), [])
            other = next((r.get("else", []) for r in rules if "else" in r), [])
            return (None, list(other if alt else first))
        if "decision_file" in b:
            d = json.load(open(Path(self.spec.workdir) / self._fmt(b["decision_file"], i)))
            return d.get("metric"), list(d.get("reveal", []))
        res = subprocess.run(self._fmt(b["metric_cmd"], i), shell=True, capture_output=True,
                             text=True, cwd=self.spec.workdir)
        metric = float(res.stdout.strip().split()[-1])
        return metric, evaluate_rules(rules, metric)

    def _sample_usage(self):
        if psutil is None:
            return
        for i, p in self.procs.items():
            u = self.usage[i]
            try:
                root = psutil.Process(p.pid)
                procs = [root] + root.children(recursive=True)
                cpu = rss = 0.0
                for q in procs:
                    pp = u["procs"].get(q.pid)
                    if pp is None:
                        u["procs"][q.pid] = q
                        q.cpu_percent(None)
                        continue
                    cpu += pp.cpu_percent(None)
                    rss += pp.memory_info().rss
                u["peak_cpu_pct"] = max(u["peak_cpu_pct"], cpu)
                u["peak_rss_gb"] = max(u["peak_rss_gb"], rss / 2**30)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    # ------------------------------------------------------------------- loop
    def run(self) -> Path:
        S = self.state
        self._t_start = time.monotonic()
        self._emit("header", policy=self.policy_name, dry_run=self.dry_run, time_scale=self.time_scale,
                   spec=self.spec.raw, samples=S.samples, capacity_units=S.cap.tolist(),
                   n_static_tasks=S.N0, lb_ref=S.lb_ref, start_unix=time.time(), **self.extra_header)
        while True:
            S.t = self._now()
            for i in S.release_until(S.t):
                inf = self._info(i)
                self._emit("visible", task=i, sample=inf["sample"], stage=inf["stage"], dynamic=False,
                           rel=float(S.rel[i]), preds=list(map(int, S.pred[i])))
            for i, p in list(self.procs.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del self.procs[i]
                u = self.usage.pop(i)
                inf = self._info(i)
                self._emit("end", task=i, sample=inf["sample"], stage=inf["stage"], exit=rc,
                           peak_cpu_pct=round(u["peak_cpu_pct"], 1), peak_rss_gb=round(u["peak_rss_gb"], 3))
                if rc != 0:
                    S.complete_task(i, S.t)
                    gone = S.cancel_sample(inf["sample_idx"], S.t)
                    self._emit("fail", task=i, sample=inf["sample"], cancelled=gone)
                    continue
                reveal = []
                metric = None
                if self.spec.stages[inf["stage"]].branch:
                    metric, reveal = self._branch(i)
                new = S.complete_task(i, S.t, reveal)
                if self.spec.stages[inf["stage"]].branch:
                    self._emit("branch", task=i, sample=inf["sample"], metric=metric, reveal=reveal, new_tasks=new)
                for d in new:
                    dinf = self._info(d)
                    self._emit("visible", task=d, sample=dinf["sample"], stage=dinf["stage"], dynamic=True,
                               rel=float(S.rel[d]), preds=list(map(int, S.pred[d])),
                               succ=list(map(int, S.succ[d])))
            D = S.dispatchable()
            while D:
                t0 = time.perf_counter()
                a = self.policy(S, D)
                self.decision_ms.append(1000 * (time.perf_counter() - t0))
                self.n_decisions += 1
                if a == WAIT:
                    break
                self._launch(int(a))
                D = S.dispatchable()
            if S.done() and not self.procs:
                break
            self._sample_usage()
            time.sleep(self.poll * (self.time_scale if self.dry_run else 1.0))
        self._emit("done", n_decisions=self.n_decisions,
                   decision_ms_mean=float(np.mean(self.decision_ms)) if self.decision_ms else 0.0,
                   decision_ms_p95=float(np.percentile(self.decision_ms, 95)) if self.decision_ms else 0.0)
        self.log.close()
        return self.run_dir / "events.jsonl"
