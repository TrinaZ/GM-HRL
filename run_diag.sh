#!/usr/bin/env bash
# Training diagnosis on VALIDATION instances only (generated k = 50..54; the
# test instances k = 0..19 are not touched). Question: does reinforcement
# learning move GM-HRL beyond its HEFT behaviour-cloning start when it gets a
# larger training budget, a response-time signal, or no cloning at all?
#   A = existing run (40 BC + 300 PPO iterations x 4 episodes, lr 1e-4)
#   B = 40 BC + 400 iterations x 12 episodes, lr 3e-4, entropy 0.02
#   C = B + response-time shaping (w5 = 0.2)
#   D = C without behaviour cloning
# Usage: JOBS=3 bash run_diag.sh [train] [eval]
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
JOBS=${JOBS:-3}; THREADS=2; PY=${PY:-python3}
DOMAINS=${DOMAINS:-"genomic video"}; N=${N:-100}
mkdir -p runs_diag results_diag logs
BIG="--iters 400 --episodes_per_iter 12 --lr 3e-4 --ent_coef 0.02 --critic_warmup 10 --eval_every 10 --threads $THREADS"

phase_train() {
  for d in $DOMAINS; do
    [ -f runs_diag/B_${d}_N$N/last.pt ] || echo "$PY scripts/train.py --variant gmhrl --domains $d --N $N --seed 0 --bc_iters 40 $BIG --out runs_diag/B_${d}_N$N > logs/diag_B_${d}_N$N.log 2>&1"
    [ -f runs_diag/C_${d}_N$N/last.pt ] || echo "$PY scripts/train.py --variant gmhrl --domains $d --N $N --seed 0 --bc_iters 40 $BIG --w5 0.2 --out runs_diag/C_${d}_N$N > logs/diag_C_${d}_N$N.log 2>&1"
    [ -f runs_diag/D_${d}_N$N/last.pt ] || echo "$PY scripts/train.py --variant gmhrl --domains $d --N $N --seed 0 --bc_iters 0 $BIG --w5 0.2 --out runs_diag/D_${d}_N$N > logs/diag_D_${d}_N$N.log 2>&1"
  done | xargs -P "$JOBS" -I{} bash -c "{}"
}

phase_eval() {
  for d in $DOMAINS; do
    m="FIFO SJF HEFT GA=runs/ga_${d}_N$N.json A=runs/gmhrl_${d}_N${N}_s0/best.pt PPO=runs/ppo_${d}_N${N}_s0/best.pt"
    for v in B C D; do [ -f runs_diag/${v}_${d}_N$N/best.pt ] && m="$m $v=runs_diag/${v}_${d}_N$N/best.pt"; done
    $PY scripts/evaluate.py --bench data/generated --domains $d --N $N --instances 50 51 52 53 54 \
        --seeds 2000 2001 2002 2003 --methods $m --ref HEFT --out results_diag/val_${d}_N$N > logs/diag_eval_${d}_N$N.log 2>&1
    tail -n 25 logs/diag_eval_${d}_N$N.log
  done
}

for p in ${*:-train eval}; do echo "=== $p ($(date)) ==="; phase_$p; done
