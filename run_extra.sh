#!/usr/bin/env bash
# Additional baselines, run AFTER run_all.sh (does not touch its outputs):
#   1) single-level GNN+PPO dispatcher (Zhang et al. 2020 style) for every (domain, N, seed);
#   2) final evaluation of all methods incl. ORHG into results/final_<domain>_N<N>,
#      reusing the clairvoyant bounds computed by run_all.sh.
# Usage: JOBS=6 THREADS=2 bash run_extra.sh [train] [eval]
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-6}; THREADS=${THREADS:-2}
DOMAINS=${DOMAINS:-"genomic video ai_ft financial"}
SCALES=${SCALES:-"50 100 200 500 1000"}
SEEDS=${SEEDS:-"0 1 2"}
ORHG_MAX_N=${ORHG_MAX_N:-500}
PY=${PY:-python3}
TRAIN_COMMON=${TRAIN_COMMON:-"--bc_iters 40 --critic_warmup 10 --iters 300 --lr 1e-4 --eval_every 10 --threads $THREADS"}
mkdir -p runs results logs
run_parallel() { xargs -P "$JOBS" -I{} bash -c "{}"; }

phase_train() {
  for n in $SCALES; do for d in $DOMAINS; do for s in $SEEDS; do
    out=runs/gnn_ppo_${d}_N${n}_s${s}
    [ -f "$out/last.pt" ] || echo "$PY scripts/train.py --variant gnn_ppo --domains $d --N $n --seed $s $TRAIN_COMMON --out $out > logs/$(basename $out).log 2>&1"
  done; done; done | run_parallel
}

phase_eval() {
  for d in $DOMAINS; do for n in $SCALES; do
    out=results/final_${d}_N${n}
    [ -f $out/summary.json ] && continue
    m="FIFO SJF HEFT Random GA=runs/ga_${d}_N${n}.json"
    [ "$n" -le "$ORHG_MAX_N" ] && m="$m ORHG"
    for s in $SEEDS; do
      for v in gmhrl ppo gnn_ppo; do
        f=runs/${v}_${d}_N${n}_s${s}/best.pt
        case $v in gmhrl) name=GM-HRL;; ppo) name=PPO;; gnn_ppo) name=GNN-PPO;; esac
        [ -f "$f" ] && m="$m ${name}_s$s=$f"
      done
    done
    b=""; [ -f results/main_${d}_N${n}/bounds.csv ] && b="--bounds_csv results/main_${d}_N${n}/bounds.csv"
    echo "$PY scripts/evaluate.py --domains $d --N $n --seeds 1000 1001 1002 --methods $m --ref GM-HRL_s0 $b --out $out > logs/final_${d}_N${n}.log 2>&1"
  done; done | run_parallel
  $PY scripts/make_tables.py --glob 'results/final_*' --out results/tables_final
}

for p in ${*:-train eval}; do echo "=== $p ($(date)) ==="; phase_$p; done
