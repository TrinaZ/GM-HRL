#!/usr/bin/env bash
# REPLICATION_SPEC.md. Evaluation only; no model is trained or tuned.
#   data      generate fresh instances k = 300..319 (v1.1 generator) for all
#             domains and N in {200, 500, 1000} into data/replication/
#   eval      all methods of the main study except ORH, realisations 1000..1002
#   decide    pre-registered replication criteria -> results_rep/replication_decision.json
#   dyngraph  Part B: GM-HRL and w/o dynamic graph on the test instances under
#             the default conditions and every condition of the robustness study
# Usage: JOBS=6 THREADS=2 bash run_replication.sh data eval decide dyngraph
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
JOBS=${JOBS:-6}; PY=${PY:-python3}
SEEDS="0 1 2"; R=runs_v2; O=results_rep; DATA=data/replication
KS="$(seq -s ' ' 300 319)"
mkdir -p $O logs
run_parallel() { xargs -P "$JOBS" -I{} bash -c "{}"; }
models_for() {  # domain N(train)
  local d=$1 n=$2 m="" v name
  for s in $SEEDS; do for v in gmhrl ppo gnn_ppo; do
    case $v in gmhrl) name=GM-HRL;; ppo) name=PPO;; gnn_ppo) name=GNN-PPO;; esac
    [ -f $R/${v}_${d}_N${n}_s${s}/best.pt ] || { echo "missing $R/${v}_${d}_N${n}_s${s}/best.pt" >&2; exit 1; }
    m="$m ${name}_s$s=$R/${v}_${d}_N${n}_s${s}/best.pt"
  done; done
  echo "$m"
}
phase_data() {
  [ -d $DATA ] && { echo "$DATA exists: instances must be generated fresh (REPLICATION_SPEC.md, section 2)"; exit 1; }
  $PY - "$DATA" "$KS" <<'PY'
import sys
sys.path.insert(0, ".")
from gmhrl.generate import write_split
root, ks = sys.argv[1], [int(k) for k in sys.argv[2].split()]
for d in ("genomic", "video", "ai_ft", "financial"):
    for n in (200, 500, 1000):
        write_split(root, d, n, ks)
    print(d, "done", flush=True)
PY
}
phase_eval() {
  for d in genomic video ai_ft financial; do for n in 200 500; do models_for $d $n > /dev/null; done; done
  for d in genomic video ai_ft financial; do for n in 200 500 1000; do
    out=$O/rep_${d}_N${n}; [ -f $out/summary.json ] && continue
    tn=$n; [ "$n" -eq 1000 ] && tn=500
    echo "$PY scripts/evaluate.py --bench $DATA --domains $d --N $n --instances $KS --seeds 1000 1001 1002 --methods FIFO SJF HEFT Random GA=runs/ga_${d}_N${n}.json $(models_for $d $tn) --ref HEFT --out $out > logs/rep_${d}_N${n}.log 2>&1"
  done; done | run_parallel
}
phase_decide() {
  $PY scripts/decide_replication.py --res $O --out $O/replication_decision.json
}
phase_dyngraph() {
  for s in $SEEDS; do for v in gmhrl no_dynamic_graph; do
    [ -f $R/${v}_genomic_N200_s$s/best.pt ] || { echo "missing $R/${v}_genomic_N200_s$s/best.pt"; exit 1; }
  done; done
  M="HEFT"
  for s in $SEEDS; do M="$M GM-HRL_s$s=$R/gmhrl_genomic_N200_s$s/best.pt no_dynamic_graph_s$s=$R/no_dynamic_graph_genomic_N200_s$s/best.pt"; done
  {
    echo "$PY scripts/evaluate.py --domains genomic --N 200 --methods $M --ref HEFT --out $O/dyn_default > logs/rep_dyn_default.log 2>&1"
    for r in 0 2.6667; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rrd_mult $r --methods $M --ref HEFT --out $O/dyn_rrd$r > logs/rep_dyn_rrd$r.log 2>&1"; done
    for rho in 0.6 0.95; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rho $rho --methods $M --ref HEFT --out $O/dyn_rho$rho > logs/rep_dyn_rho$rho.log 2>&1"; done
    echo "$PY scripts/evaluate.py --domains genomic --N 200 --cap_scale 0.7 --methods $M --ref HEFT --out $O/dyn_cap0.7 > logs/rep_dyn_cap0.7.log 2>&1"
  } | run_parallel
  $PY scripts/decide_replication.py --res $O --dyngraph --out $O/dyngraph_stats.json
}
for p in "$@"; do echo "=== $p ($(date)) ==="; phase_$p; done
echo "=== done ($(date)) ==="
