#!/usr/bin/env bash
# Second round of experiments with the training budget selected on the
# VALIDATION set (run_diag.sh, configuration B):
#   40 BC iterations, then 400 PPO iterations x 12 episodes, lr 3e-4, entropy 0.02.
# The same budget is used for every learning-based method (GM-HRL, PPO, GNN-PPO,
# ablations, kappa variants). Instances N <= 500 are trained directly; N = 1000
# is evaluated zero-shot with the N = 500 policies (size transfer, R3-6).
# Writes to runs_v2/ and results_v2/; the first round (runs/, results/) is kept.
# Clairvoyant bounds are reused from results/main_*/bounds.csv.
# Usage: JOBS=6 THREADS=2 bash run_v2.sh [train] [ablation] [pareto] [eval] [eval_ablation] [robustness] [transfer]
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
JOBS=${JOBS:-6}; THREADS=${THREADS:-2}; PY=${PY:-python3}
DOMAINS=${DOMAINS:-"genomic video ai_ft financial"}
TRAIN_SCALES=${TRAIN_SCALES:-"50 100 200 500"}
SEEDS=${SEEDS:-"0 1 2"}
ORHG_MAX_N=${ORHG_MAX_N:-500}
R=runs_v2; O=results_v2
TRAIN_COMMON=${TRAIN_COMMON:-"--bc_iters 40 --critic_warmup 10 --iters 400 --episodes_per_iter 12 --lr 3e-4 --ent_coef 0.02 --eval_every 10 --threads $THREADS"}
mkdir -p $R $O logs
run_parallel() { xargs -P "$JOBS" -I{} bash -c "{}"; }

train_cmd() {  # variant domain N seed tag extra...
  local v=$1 d=$2 n=$3 s=$4 tag=$5; shift 5
  local out=$R/${v}${tag}_${d}_N${n}_s${s}
  [ -f "$out/last.pt" ] || echo "$PY scripts/train.py --variant $v --domains $d --N $n --seed $s $TRAIN_COMMON $* --out $out > logs/v2_$(basename $out).log 2>&1"
}

phase_train() {   # largest first so the long runs start early
  for n in $(echo $TRAIN_SCALES | tr ' ' '\n' | sort -rn); do for d in $DOMAINS; do for s in $SEEDS; do
    for v in gmhrl ppo gnn_ppo; do train_cmd $v $d $n $s ""; done
  done; done; done | run_parallel
}
phase_ablation() {
  for v in no_hierarchy no_gnn no_dynamic_graph; do for s in $SEEDS; do train_cmd $v genomic 200 $s ""; done; done | run_parallel
}
phase_pareto() {
  for k in 0.3 3 10; do train_cmd gmhrl genomic 200 0 "_kappa$k" --kappa $k; done | run_parallel
}

models_for() {  # domain N(train) -> method specs
  local d=$1 n=$2 m="" v name
  for s in $SEEDS; do for v in gmhrl ppo gnn_ppo; do
    case $v in gmhrl) name=GM-HRL;; ppo) name=PPO;; gnn_ppo) name=GNN-PPO;; esac
    [ -f $R/${v}_${d}_N${n}_s${s}/best.pt ] && m="$m ${name}_s$s=$R/${v}_${d}_N${n}_s${s}/best.pt"
  done; done
  echo "$m"
}

phase_eval() {
  for d in $DOMAINS; do for n in ${EVAL_SCALES:-50 100 200 500 1000}; do
    out=$O/main_${d}_N${n}; [ -f $out/summary.json ] && continue
    tn=$n; [ "$n" -eq 1000 ] && tn=500           # N=1000: policies trained at N=500
    m="FIFO SJF HEFT Random GA=runs/ga_${d}_N${n}.json"
    [ "$n" -le "$ORHG_MAX_N" ] && m="$m ORHG"
    b=""; [ -f results/main_${d}_N${n}/bounds.csv ] && b="--bounds_csv results/main_${d}_N${n}/bounds.csv"
    echo "$PY scripts/evaluate.py --domains $d --N $n --seeds 1000 1001 1002 --methods $m $(models_for $d $tn) --ref HEFT $b --out $out > logs/v2_eval_${d}_N${n}.log 2>&1"
  done; done | run_parallel
  $PY scripts/make_tables.py --glob "$O/main_*" --out $O/tables_main || true
}

phase_eval_ablation() {
  m="HEFT GA=runs/ga_genomic_N200.json"
  for s in $SEEDS; do
    m="$m GM-HRL_s$s=$R/gmhrl_genomic_N200_s$s/best.pt"
    for v in no_hierarchy no_gnn no_dynamic_graph; do m="$m ${v}_s$s=$R/${v}_genomic_N200_s$s/best.pt"; done
  done
  for k in 0.3 3 10; do m="$m kappa${k}=$R/gmhrl_kappa${k}_genomic_N200_s0/best.pt"; done
  b=""; [ -f results/main_genomic_N200/bounds.csv ] && b="--bounds_csv results/main_genomic_N200/bounds.csv"
  $PY scripts/evaluate.py --domains genomic --N 200 --seeds 1000 1001 1002 --methods $m --ref HEFT $b \
      --out $O/ablation_genomic_N200 > logs/v2_eval_ablation.log 2>&1
}

phase_robustness() {   # all three seeds, no retraining
  M="FIFO SJF HEFT GA=runs/ga_genomic_N200.json $(models_for genomic 200)"
  {
    for r in 0 2.6667; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rrd_mult $r --methods $M --ref HEFT --out $O/robust_rrd$r > logs/v2_robust_rrd$r.log 2>&1"; done
    for rho in 0.6 0.95; do echo "$PY scripts/evaluate.py --domains genomic --N 200 --rho $rho --methods $M --ref HEFT --out $O/robust_rho$rho > logs/v2_robust_rho$rho.log 2>&1"; done
    echo "$PY scripts/evaluate.py --domains genomic --N 200 --cap_scale 0.7 --methods $M --ref HEFT --out $O/robust_cap0.7 > logs/v2_robust_cap0.7.log 2>&1"
  } | run_parallel
}

phase_transfer() {     # zero-shot: genomic-trained policies on other domains; N=100 policies on larger N
  {
    for d in video ai_ft financial; do
      m="HEFT GA=runs/ga_${d}_N200.json"
      for s in $SEEDS; do m="$m GM-HRL-genomic_s$s=$R/gmhrl_genomic_N200_s$s/best.pt GM-HRL-in-domain_s$s=$R/gmhrl_${d}_N200_s$s/best.pt"; done
      echo "$PY scripts/evaluate.py --domains $d --N 200 --methods $m --ref HEFT --out $O/xdomain_$d > logs/v2_xdomain_$d.log 2>&1"
    done
    for n in 200 500 1000; do
      m="HEFT GA=runs/ga_genomic_N${n}.json"
      for s in $SEEDS; do m="$m GM-HRL-N100_s$s=$R/gmhrl_genomic_N100_s$s/best.pt"; done
      [ "$n" -le 500 ] && for s in $SEEDS; do m="$m GM-HRL-in-size_s$s=$R/gmhrl_genomic_N${n}_s$s/best.pt"; done
      echo "$PY scripts/evaluate.py --domains genomic --N $n --methods $m --ref HEFT --out $O/size_N$n > logs/v2_size_N$n.log 2>&1"
    done
  } | run_parallel
}

for p in ${*:-train ablation pareto eval eval_ablation robustness transfer}; do echo "=== $p ($(date)) ==="; phase_$p; done
echo "=== done ($(date)) ==="
