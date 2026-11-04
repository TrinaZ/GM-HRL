#!/usr/bin/env bash
# GM-HRL+R (capacity reservation), RESERVATION_SPEC.md. Phases:
#   val_train  train gmhrl_res, genomic + video, N = 200, seeds 0-2 (round-2 budget)
#   val_eval   validation instances k = 50..54, seeds 2000..2003, vs round-2 GM-HRL
#   decide     pre-registered adoption decision -> results_v2/reserve_decision.json
#   test200    test-set evaluation of the six N = 200 models (reported in either case)
#   full_train / full_eval   only if the decision is ADOPT (refuses otherwise)
# Usage: JOBS=6 THREADS=2 bash run_reserve.sh val_train val_eval decide test200
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
JOBS=${JOBS:-6}; THREADS=${THREADS:-2}; PY=${PY:-python3}
SEEDS="0 1 2"; R=runs_v2; O=results_v2
TRAIN_COMMON="--bc_iters 40 --critic_warmup 10 --iters 400 --episodes_per_iter 12 --lr 3e-4 --ent_coef 0.02 --eval_every 10 --threads $THREADS"
mkdir -p $R $O logs
run_parallel() { xargs -P "$JOBS" -I{} bash -c "{}"; }
train_cmd() {  # variant domain N seed tag extra...
  local v=$1 d=$2 n=$3 s=$4 tag=$5; shift 5
  local out=$R/${v}${tag}_${d}_N${n}_s${s}
  [ -f "$out/last.pt" ] || echo "$PY scripts/train.py --variant $v --domains $d --N $n --seed $s $TRAIN_COMMON $* --out $out > logs/res_$(basename $out).log 2>&1"
}
pair_models() {  # domain N(train) -> control and reservation models
  local d=$1 n=$2 m=""
  for s in $SEEDS; do
    m="$m GM-HRL_s$s=$R/gmhrl_${d}_N${n}_s$s/best.pt"
    m="$m GM-HRL+R_s$s=$R/gmhrl_res_${d}_N${n}_s$s/best.pt"
  done
  echo "$m"
}
require_adopt() {
  $PY -c "import json,sys; sys.exit(0 if json.load(open('$O/reserve_decision.json'))['adopt'] else 1)" \
    || { echo "decision is not ADOPT: full phases are not run (RESERVATION_SPEC.md, section 5)"; exit 1; }
}

phase_val_train() {
  for d in genomic video; do for s in $SEEDS; do train_cmd gmhrl_res $d 200 $s ""; done; done | run_parallel
}
phase_val_eval() {
  for d in genomic video; do
    echo "$PY scripts/evaluate.py --bench data/generated --domains $d --N 200 --instances 50 51 52 53 54 --seeds 2000 2001 2002 2003 --methods HEFT $(pair_models $d 200) --ref HEFT --res_log --out $O/val_res_${d}_N200 > logs/res_val_${d}.log 2>&1"
  done | run_parallel
}
phase_decide() {
  $PY scripts/decide_reserve.py --res $O --out $O/reserve_decision.json
}
phase_test200() {
  for d in genomic video; do
    b=""; [ -f results/main_${d}_N200/bounds.csv ] && b="--bounds_csv results/main_${d}_N200/bounds.csv"
    echo "$PY scripts/evaluate.py --domains $d --N 200 --seeds 1000 1001 1002 --methods HEFT $(pair_models $d 200) --ref HEFT $b --res_log --out $O/res_main_${d}_N200 > logs/res_test_${d}.log 2>&1"
  done | run_parallel
}
phase_full_train() {
  require_adopt
  {
    for n in 500 100 50; do for d in genomic video ai_ft financial; do for s in $SEEDS; do train_cmd gmhrl_res $d $n $s ""; done; done; done
    for d in ai_ft financial; do for s in $SEEDS; do train_cmd gmhrl_res $d 200 $s ""; done; done
    for v in no_gnn_res no_dynamic_graph_res; do for s in $SEEDS; do train_cmd $v genomic 200 $s ""; done; done
    for k in 0.3 3 10; do train_cmd gmhrl_res genomic 200 0 "_kappa$k" --kappa $k; done
  } | run_parallel
}
phase_full_eval() {
  require_adopt
  {
    for d in genomic video ai_ft financial; do for n in 50 100 200 500 1000; do
      out=$O/res_main_${d}_N${n}; [ -f $out/summary.json ] && continue
      tn=$n; [ "$n" -eq 1000 ] && tn=500
      b=""; [ -f results/main_${d}_N${n}/bounds.csv ] && b="--bounds_csv results/main_${d}_N${n}/bounds.csv"
      echo "$PY scripts/evaluate.py --domains $d --N $n --seeds 1000 1001 1002 --methods HEFT $(pair_models $d $tn) --ref HEFT $b --res_log --out $out > logs/res_eval_${d}_N${n}.log 2>&1"
    done; done
    m="HEFT GA=runs/ga_genomic_N200.json"
    for s in $SEEDS; do
      m="$m GM-HRL+R_s$s=$R/gmhrl_res_genomic_N200_s$s/best.pt GM-HRL_s$s=$R/gmhrl_genomic_N200_s$s/best.pt"
      m="$m no_hierarchy_s$s=$R/no_hierarchy_genomic_N200_s$s/best.pt"
      for v in no_gnn_res no_dynamic_graph_res; do m="$m ${v}_s$s=$R/${v}_genomic_N200_s$s/best.pt"; done
    done
    for k in 0.3 3 10; do m="$m kappa${k}=$R/gmhrl_res_kappa${k}_genomic_N200_s0/best.pt"; done
    b=""; [ -f results/main_genomic_N200/bounds.csv ] && b="--bounds_csv results/main_genomic_N200/bounds.csv"
    echo "$PY scripts/evaluate.py --domains genomic --N 200 --seeds 1000 1001 1002 --methods $m --ref HEFT $b --out $O/res_ablation_genomic_N200 > logs/res_eval_ablation.log 2>&1"
    M="HEFT $(pair_models genomic 200)"
    for r in 0 2.6667; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rrd_mult $r --methods $M --ref HEFT --out $O/res_robust_rrd$r > logs/res_robust_rrd$r.log 2>&1"; done
    for rho in 0.6 0.95; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rho $rho --methods $M --ref HEFT --out $O/res_robust_rho$rho > logs/res_robust_rho$rho.log 2>&1"; done
    echo "$PY scripts/evaluate.py --domains genomic --N 200 --cap_scale 0.7 --methods $M --ref HEFT --out $O/res_robust_cap0.7 > logs/res_robust_cap0.7.log 2>&1"
    for d in video ai_ft financial; do
      m="HEFT"; for s in $SEEDS; do m="$m GM-HRL+R-genomic_s$s=$R/gmhrl_res_genomic_N200_s$s/best.pt GM-HRL+R-in-domain_s$s=$R/gmhrl_res_${d}_N200_s$s/best.pt"; done
      echo "$PY scripts/evaluate.py --domains $d --N 200 --methods $m --ref HEFT --out $O/res_xdomain_$d > logs/res_xdomain_$d.log 2>&1"
    done
    for n in 200 500 1000; do
      m="HEFT"; for s in $SEEDS; do m="$m GM-HRL+R-N100_s$s=$R/gmhrl_res_genomic_N100_s$s/best.pt"; done
      [ "$n" -le 500 ] && for s in $SEEDS; do m="$m GM-HRL+R-in-size_s$s=$R/gmhrl_res_genomic_N${n}_s$s/best.pt"; done
      echo "$PY scripts/evaluate.py --domains genomic --N $n --methods $m --ref HEFT --out $O/res_size_N$n > logs/res_size_N$n.log 2>&1"
    done
  } | run_parallel
}

for p in "$@"; do echo "=== $p ($(date)) ==="; phase_$p; done
echo "=== done ($(date)) ==="
