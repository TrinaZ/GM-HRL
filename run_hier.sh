#!/usr/bin/env bash
# MD2POS-MW: does the hierarchy help when workflows of different urgency compete?
# Pre-registered in HIERARCHY_SPEC.md. Phases:
#   data       generate all MD2POS-MW batches (serially, before parallel jobs)
#   ga         evolve the GA rule on MD2POS-MW training batches (genomic, video)
#   train      PPO, GNN-PPO, GM-HRL (round-2 architecture) and GM-HRL v2,
#              genomic + video, K = 12, s = 0.2, seeds 0-2, configuration B
#   val_eval   validation batches k = 50..54 x realisations 2000..2003 (decision set)
#   decide     M1, M2, H1, H2 -> results_hier/hier_decision.json
#   test       test batches k = 0..19 x realisations 1000..1002 (reported in any case)
#   secondary  test batches with K = 30 (s = 0.2) and K = 12 with s = 0.1 / 0.4
#              (main-setting policies, no retraining; descriptive)
# Usage: JOBS=6 THREADS=2 bash run_hier.sh data ga train val_eval decide test secondary
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
JOBS=${JOBS:-6}; THREADS=${THREADS:-2}; PY=${PY:-python3}
SEEDS="0 1 2"; DOMS="genomic video"
R=${R:-runs_hier}; O=${O:-results_hier}; DATA=${DATA:-data/generated}
# the three overrides below exist only for code smoke tests; the study uses the defaults
VAL_K=${VAL_K:-"50 51 52 53 54"}; TEST_K=${TEST_K:-"$(seq -s ' ' 0 19)"}
BUDGET=${BUDGET:-"--bc_iters 40 --critic_warmup 10 --iters 400 --episodes_per_iter 12"}
N_TRAIN=${N_TRAIN:-100}; GA_ARGS=${GA_ARGS:-""}
MW="--mw_K 12 --stat_share 0.2"
TRAIN_COMMON="$BUDGET --lr 3e-4 --ent_coef 0.02 --eval_every 10 --threads $THREADS --n_train $N_TRAIN --expert wedf --data $DATA"
VARIANTS="ppo:PPO gnn_ppo:GNN-PPO gmhrl:GM-HRL gmhrl_v2:GM-HRL-v2"
mkdir -p $R $O logs
run_parallel() { xargs -P "$JOBS" -I{} bash -c "{}"; }

methods() {  # domain [diag] -> evaluation method list
  local d=$1 diag=${2:-} m="FIFO HEFT EDF WEDF GA=$R/ga_${d}_MW.json" v n s
  for vn in $VARIANTS; do
    v=${vn%%:*}; n=${vn##*:}
    for s in $SEEDS; do m="$m ${n}_s$s=$R/${v}_${d}_s$s/best.pt"; done
  done
  if [ -n "$diag" ]; then
    for s in $SEEDS; do
      m="$m GM-HRL-v2-shuffle_s$s=$R/gmhrl_v2_${d}_s$s/best.pt@shuffle"
      m="$m GM-HRL-v2-random_s$s=$R/gmhrl_v2_${d}_s$s/best.pt@random"
    done
  fi
  echo "$m"
}

phase_data() {
  $PY - "$DATA" "$N_TRAIN" "$VAL_K" "$TEST_K" <<'EOF'
import sys
sys.path.insert(0, ".")
from gmhrl.multiwf import write_split_mw
data, n_train = sys.argv[1], int(sys.argv[2])
val = [int(k) for k in sys.argv[3].split()]
test = [int(k) for k in sys.argv[4].split()]
for d in ("genomic", "video"):
    write_split_mw(data, d, 12, 0.2, list(range(100, 100 + n_train)) + list(range(60, 65)) + val + test)
    write_split_mw(data, d, 30, 0.2, test)
    for s in (0.1, 0.4):
        write_split_mw(data, d, 12, s, test)
    print(d, "done", flush=True)
EOF
}
phase_ga() {
  for d in $DOMS; do
    [ -f $R/ga_${d}_MW.json ] || echo "$PY scripts/train_ga.py --domains $d $MW --data $DATA $GA_ARGS --out $R/ga_${d}_MW.json > logs/hier_ga_$d.log 2>&1"
  done | run_parallel
}
phase_train() {
  for vn in $VARIANTS; do v=${vn%%:*}; for d in $DOMS; do for s in $SEEDS; do
    out=$R/${v}_${d}_s$s
    [ -f "$out/last.pt" ] || echo "$PY scripts/train.py --variant $v --domains $d --seed $s $MW $TRAIN_COMMON --out $out > logs/hier_$(basename $out).log 2>&1"
  done; done; done | run_parallel
}
phase_val_eval() {
  for d in $DOMS; do
    echo "$PY scripts/evaluate.py $MW --data $DATA --domains $d --instances $VAL_K --seeds 2000 2001 2002 2003 --methods $(methods $d diag) --ref GM-HRL-v2_s0 --goal_measure --out $O/val_$d > logs/hier_val_$d.log 2>&1"
  done | run_parallel
}
phase_decide() {
  $PY scripts/decide_hier.py --res $O --out $O/hier_decision.json
}
phase_test() {
  for d in $DOMS; do
    echo "$PY scripts/evaluate.py $MW --data $DATA --domains $d --instances $TEST_K --seeds 1000 1001 1002 --methods $(methods $d diag) --ref GM-HRL-v2_s0 --goal_measure --out $O/test_$d > logs/hier_test_$d.log 2>&1"
  done | run_parallel
}
phase_secondary() {
  for d in $DOMS; do
    echo "$PY scripts/evaluate.py --mw_K 30 --stat_share 0.2 --data $DATA --domains $d --instances $TEST_K --seeds 1000 1001 1002 --methods $(methods $d) --ref GM-HRL-v2_s0 --goal_measure --out $O/test_K30_$d > logs/hier_K30_$d.log 2>&1"
    for s in 0.1 0.4; do
      echo "$PY scripts/evaluate.py --mw_K 12 --stat_share $s --data $DATA --domains $d --instances $TEST_K --seeds 1000 1001 1002 --methods $(methods $d) --ref GM-HRL-v2_s0 --goal_measure --out $O/test_s${s}_$d > logs/hier_s${s}_$d.log 2>&1"
    done
  done | run_parallel
}

for p in "$@"; do echo "=== $p ($(date)) ==="; phase_$p; done
echo "=== done ($(date)) ==="
