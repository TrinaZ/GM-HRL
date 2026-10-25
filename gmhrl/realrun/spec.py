"""Pipeline specification for real runs (YAML or JSON).

Example: see realrun/example_wgs.yaml. Semantics:

* Each sample instantiates every non-dynamic stage as one task, with the
  static precedences given by ``after``.
* A stage with ``branch`` is a runtime-revealed-dependency branch point. When
  its task finishes, the branch decision is obtained either from
  ``metric_cmd`` (prints a number; ``rules`` map it to stages to reveal) or
  from ``decision_file`` (JSON written by the stage itself:
  {"metric": 0.91, "reveal": ["gatk"]}). Revealed (``dynamic: true``) stages
  are inserted between the branch task and its static successors, exactly as
  in the simulator.
* Resource requests are converted to model units with ``model_units``
  (the policies were trained with a capacity of 4 units per resource).
"""
from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

RES = ("cpu", "ram", "gpu", "disk")
_OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}


@dataclass
class Stage:
    name: str
    model_type: str
    cmd: str
    request: Dict[str, float]
    est_seconds: float
    after: List[str] = field(default_factory=list)
    dynamic: bool = False
    branch: Optional[Dict] = None

    @property
    def rrd_prob(self) -> float:
        return float(self.branch.get("rrd_prob", 0.0)) if self.branch else 0.0


@dataclass
class PipelineSpec:
    name: str
    stages: Dict[str, Stage]
    model_types: List[str]
    model_units: Dict[str, float]
    capacity: Dict[str, float]
    instances_per_stage: int
    sla_seconds: float
    workdir: str
    raw: Dict

    def demand_units(self, st: Stage):
        return [st.request.get(r, 0.0) / self.model_units[r] for r in RES]

    def capacity_units(self):
        return [self.capacity[r] / self.model_units[r] for r in RES]

    def static_stages(self):
        return [s for s in self.stages.values() if not s.dynamic]

    def static_successors(self, name: str) -> List[str]:
        return [s.name for s in self.static_stages() if name in s.after]


def evaluate_rules(rules: List[Dict], metric: float) -> List[str]:
    for r in rules:
        if "if" in r:
            m = re.match(r"\s*(>=|<=|==|>|<)\s*([-+0-9.eE]+)\s*$", str(r["if"]))
            if not m:
                raise ValueError(f"bad rule condition {r['if']!r}")
            if _OPS[m.group(1)](metric, float(m.group(2))):
                return list(r.get("reveal", []))
        elif "else" in r:
            return list(r["else"])
    return []


def load_spec(path: str | Path) -> PipelineSpec:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    model_types = list(raw.get("model_types", ["BWA-MEM", "SAMtools-sort", "SAMtools-idx", "FreeBayes",
                                               "GATK-HC", "bcftools", "FastQC", "MultiQC"]))
    stages = {}
    for s in raw["stages"]:
        if s["model_type"] not in model_types:
            raise ValueError(f"stage {s['name']}: model_type {s['model_type']!r} not in {model_types}")
        stages[s["name"]] = Stage(
            name=s["name"], model_type=s["model_type"], cmd=s.get("cmd", ""),
            request={r: float(s.get(r, 0.0)) for r in RES}, est_seconds=float(s["est_seconds"]),
            after=list(s.get("after", [])), dynamic=bool(s.get("dynamic", False)), branch=s.get("branch"))
    for st in stages.values():
        for a in st.after:
            if a not in stages:
                raise ValueError(f"stage {st.name}: unknown predecessor {a}")
        if st.branch:
            names = set()
            for r in st.branch.get("rules", []):
                names |= set(r.get("reveal", [])) | set(r.get("else", []))
            for n in names:
                if n not in stages or not stages[n].dynamic:
                    raise ValueError(f"branch of {st.name} reveals {n!r}, which must be a dynamic stage")
    spec = PipelineSpec(
        name=raw.get("name", path.stem), stages=stages, model_types=model_types,
        model_units={r: float(raw["model_units"][r]) for r in RES},
        capacity={r: float(raw["capacity"][r]) for r in RES},
        instances_per_stage=int(raw.get("instances_per_stage", 4)),
        sla_seconds=float(raw["sla_seconds"]), workdir=str(raw.get("workdir", "./realrun_work")), raw=raw)
    for st in stages.values():
        d = spec.demand_units(st)
        c = spec.capacity_units()
        if any(d[k] > c[k] + 1e-9 for k in range(4)):
            raise ValueError(f"stage {st.name} requests more than the system capacity")
    return spec
