#!/usr/bin/env bash
# Full experiment pipeline for the revised manuscript.
# Usage:  JOBS=6 THREADS=2 bash run_all.sh [phase ...]
#   phases: ga main ablation pareto eval eval_ablation robustness transfer   (default: all, in order)
# Every phase is restartable: a training run is skipped if its last.pt exists.
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-6}
THREADS=${THREADS:-2}
DOMAINS=${DOMAINS:-"genomic video ai_ft financial"}
SCALES=${SCALES:-"50 100 200 500 1000"}
SEEDS=${SEEDS:-"0 1 2"}
BOUND_MAX_N=${BOUND_MAX_N:-200}
PY=${PY:-python3}
mkdir -p runs results logs

TRAIN_COMMON=${TRAIN_COMMON:-"--bc_iters 40 --critic_warmup 10 --iters 300 --lr 1e-4 --eval_every 10 --threads $THREADS"}

run_parallel() { xargs -P "$JOBS" -I{} bash -c "{}"; }

phase_ga() {
  for d in $DOMAINS; do for n in $SCALES; do
    f=runs/ga_${d}_N${n}.json
    [ -f "$f" ] || echo "$PY scripts/train_ga.py --domains $d --N $n --out $f > logs/ga_${d}_N${n}.log 2>&1"
  done; done | run_parallel
}

train_cmd() {  # variant domain N seed extra-tag extra-args
  local v=$1 d=$2 n=$3 s=$4 tag=$5; shift 5
  local out=runs/${v}${tag}_${d}_N${n}_s${s}
  [ -f "$out/last.pt" ] || echo "$PY scripts/train.py --variant $v --domains $d --N $n --seed $s $TRAIN_COMMON $* --out $out > logs/$(basename $out).log 2>&1"
}

phase_main() {
  for n in $SCALES; do for d in $DOMAINS; do for s in $SEEDS; do
    train_cmd gmhrl $d $n $s ""
    train_cmd ppo   $d $n $s ""
  done; done; done | run_parallel
}

phase_ablation() {
  for v in no_hierarchy no_gnn no_dynamic_graph; do for s in $SEEDS; do
    train_cmd $v genomic 200 $s ""
  done; done | run_parallel
}

phase_pareto() {
  for k in 0.3 3 10; do
    train_cmd gmhrl genomic 200 0 "_kappa$k" --kappa $k
  done | run_parallel
}

models_for() {  # domain N -> method specs for evaluate.py
  local d=$1 n=$2 m=""
  for s in $SEEDS; do
    [ -f runs/gmhrl_${d}_N${n}_s${s}/best.pt ] && m="$m GM-HRL_s$s=runs/gmhrl_${d}_N${n}_s${s}/best.pt"
    [ -f runs/ppo_${d}_N${n}_s${s}/best.pt ]   && m="$m PPO_s$s=runs/ppo_${d}_N${n}_s${s}/best.pt"
  done
  echo "$m"
}

phase_eval() {
  for d in $DOMAINS; do for n in $SCALES; do
    out=results/main_${d}_N${n}
    [ -f $out/summary.json ] && continue
    b=""; [ "$n" -le "$BOUND_MAX_N" ] && b="--bound --bound_time 300 --bound_workers $THREADS"
    echo "$PY scripts/evaluate.py --domains $d --N $n --seeds 1000 1001 1002 --methods FIFO SJF HEFT Random GA=runs/ga_${d}_N${n}.json $(models_for $d $n) --ref GM-HRL_s0 $b --out $out > logs/eval_${d}_N${n}.log 2>&1"
  done; done | run_parallel
}

phase_eval_ablation() {
  m="HEFT"
  for s in $SEEDS; do
    m="$m GM-HRL_s$s=runs/gmhrl_genomic_N200_s$s/best.pt"
    for v in no_hierarchy no_gnn no_dynamic_graph; do m="$m ${v}_s$s=runs/${v}_genomic_N200_s$s/best.pt"; done
  done
  for k in 0.3 3 10; do m="$m kappa${k}=runs/gmhrl_kappa${k}_genomic_N200_s0/best.pt"; done
  $PY scripts/evaluate.py --domains genomic --N 200 --seeds 1000 1001 1002 --methods $m --ref GM-HRL_s0 \
      --out results/ablation_genomic_N200 > logs/eval_ablation.log 2>&1
}

phase_robustness() {
  # policies trained at the default conditions, evaluated under shifted conditions
  M="FIFO SJF HEFT GA=runs/ga_genomic_N200.json GM-HRL_s0=runs/gmhrl_genomic_N200_s0/best.pt PPO_s0=runs/ppo_genomic_N200_s0/best.pt"
  {
    for r in 0 2.6667; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rrd_mult $r --methods $M --ref GM-HRL_s0 --out results/robust_rrd$r > logs/robust_rrd$r.log 2>&1"; done
    for rho in 0.6 0.95; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rho $rho --methods $M --ref GM-HRL_s0 --out results/robust_rho$rho > logs/robust_rho$rho.log 2>&1"; done
    for c in 0.7; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --cap_scale $c --methods $M --ref GM-HRL_s0 --out results/robust_cap$c > logs/robust_cap$c.log 2>&1"; done
  } | run_parallel
}

phase_transfer() {
  {
    # zero-shot cross-domain: genomic-trained policy on the other domains
    for d in video ai_ft financial; do
      echo "$PY scripts/evaluate.py --domains $d --N 200 --methods HEFT GA=runs/ga_${d}_N200.json GM-HRL-genomic=runs/gmhrl_genomic_N200_s0/best.pt GM-HRL-in-domain=runs/gmhrl_${d}_N200_s0/best.pt --ref GM-HRL-genomic --out results/xdomain_$d > logs/xdomain_$d.log 2>&1"
    done
    # zero-shot size transfer: N=100-trained policy on larger instances
    for n in 200 500 1000; do
      echo "$PY scripts/evaluate.py --domains genomic --N $n --methods HEFT GM-HRL-N100=runs/gmhrl_genomic_N100_s0/best.pt GM-HRL-in-size=runs/gmhrl_genomic_N${n}_s0/best.pt --ref GM-HRL-N100 --out results/size_N$n > logs/size_N$n.log 2>&1"
    done
  } | run_parallel
}

PHASES=${*:-"ga main ablation pareto eval eval_ablation robustness transfer"}
for p in $PHASES; do
  echo "=== phase $p ($(date)) ==="
  phase_$p
done
echo "=== done ($(date)) ==="
