#!/usr/bin/env bash
# Amendment v1.1 (HIERARCHY_SPEC_AMENDMENT_v1.1.md): descriptive load sweep.
# Evaluation only: the K = 12 policies trained at rho = 0.8 by run_hier.sh are
# evaluated, without retraining, at rho = 0.7, 0.9, 0.95 on the test batches
# k = 0..19 x realisations 1000..1002, all methods of the main comparison.
# Run after `run_hier.sh ... train` has finished:
#   JOBS=6 THREADS=2 bash run_hier_load.sh
# (a separate script, so that the running run_hier.sh is not modified)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
JOBS=${JOBS:-6}; PY=${PY:-python3}
SEEDS="0 1 2"; DOMS="genomic video"; RHOS="0.7 0.9 0.95"
R=${R:-runs_hier}; O=${O:-results_hier}; DATA=${DATA:-data/generated}
TEST_K=${TEST_K:-"$(seq -s ' ' 0 19)"}
VARIANTS="ppo:PPO gnn_ppo:GNN-PPO gmhrl:GM-HRL gmhrl_v2:GM-HRL-v2"
mkdir -p $O logs
methods() {
  local d=$1 m="FIFO HEFT EDF WEDF GA=$R/ga_${d}_MW.json" v n s
  for vn in $VARIANTS; do
    v=${vn%%:*}; n=${vn##*:}
    for s in $SEEDS; do
      [ -f "$R/${v}_${d}_s$s/last.pt" ] || { echo "missing $R/${v}_${d}_s$s/last.pt: training not finished" >&2; exit 1; }
      m="$m ${n}_s$s=$R/${v}_${d}_s$s/best.pt"
    done
  done
  echo "$m"
}
for d in $DOMS; do methods $d > /dev/null; done
for rho in $RHOS; do for d in $DOMS; do
  out=$O/test_rho${rho}_$d
  [ -f $out/summary.json ] && continue
  echo "$PY scripts/evaluate.py --mw_K 12 --stat_share 0.2 --rho $rho --data $DATA --domains $d --instances $TEST_K --seeds 1000 1001 1002 --methods $(methods $d) --ref GM-HRL-v2_s0 --goal_measure --out $out > logs/hier_rho${rho}_$d.log 2>&1"
done; done | xargs -P "$JOBS" -I{} bash -c "{}"
echo "=== load sweep done ($(date)) ==="
