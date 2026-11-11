#!/usr/bin/env bash
# Real-pipeline experiment, plan B. Usage (from the repository root):
#   bash realrun/wgs_chr22/run_real.sh dry      # rehearsal with sleeps, ~2 min, no data touched
#   bash realrun/wgs_chr22/run_real.sh real     # FIFO, HEFT, GM-HRL in this fixed order (~10-12 h)
#   bash realrun/wgs_chr22/run_real.sh analyze  # summary + CP-SAT bound (1 h per run)
# A finished scheduler run (events.jsonl with an 'end' record) is not repeated.
set -euo pipefail
MODE=${1:?dry|real|analyze}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${W:-/mnt/e/ForPhD/Data/wgs_chr22}
GAP=${GAP:-240}; SEED=${SEED:-2026}
ORDER=${ORDER:-"fifo heft gmhrl"}; TAG=${TAG:-real}
CKPT=${CKPT:-$REPO/runs_v2/gmhrl_genomic_N500_s0/best.pt}
RUNS=${RUNS:-$REPO/realruns}
mkdir -p "$RUNS"
export W OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BATCH=$RUNS/batch.tsv
[ -f "$BATCH" ] || python3 "$REPO/realrun/wgs_chr22/make_batch.py" "$W" "$GAP" "$SEED" "$BATCH"

spec_for() {   # policy -> spec file with its own work directory
  local p=$1 wd=$2 f=$RUNS/spec_$1.yaml
  sed -e "s#WORKDIR#$wd#" -e "s#REPO#$REPO#g" "$REPO/realrun/wgs_chr22/wgs_chr22.yaml" > "$f"
  echo "$f"
}
done_run() { [ -f "$1/events.jsonl" ] && grep -q '"ev": "done"' "$1/events.jsonl"; }

case $MODE in
dry)
  for p in fifo heft gmhrl; do
    rd=$RUNS/dry_$p; rm -rf "$rd"
    spec=$(spec_for "dry_$p" "$RUNS/dry_work_$p")
    python3 "$REPO/scripts/realrun.py" run --spec "$spec" --samples "$BATCH" --policy $p \
      $( [ $p = gmhrl ] && echo --ckpt "$CKPT" ) --run_dir "$rd" --dry_run --time_scale 0.005
  done
  python3 "$REPO/scripts/realrun.py" analyze "$RUNS"/dry_fifo "$RUNS"/dry_heft "$RUNS"/dry_gmhrl --out "$RUNS/dry_summary" ;;
real)
  for p in $ORDER; do
    rd=$RUNS/${TAG}_$p
    if done_run "$rd"; then echo "skip $p: finished"; continue; fi
    wd=$W/run_$p; [ "$TAG" = real ] || wd=$W/run_${TAG}_$p
    rm -rf "$rd" "$wd"
    spec=$(spec_for "${TAG}_$p" "$wd")
    echo "=== $p start $(date)"
    python3 "$REPO/scripts/realrun.py" run --spec "$spec" --samples "$BATCH" --policy $p \
      $( [ $p = gmhrl ] && echo --ckpt "$CKPT" ) --run_dir "$rd"
    echo "=== $p end $(date)"
    rm -f "$wd"/*/*.dedup.bam "$wd"/*/*.dedup.bam.bai   # keep VCFs/QC, free ~25 GB
  done ;;
analyze)
  python3 "$REPO/scripts/realrun.py" analyze "$RUNS"/${TAG}_fifo "$RUNS"/${TAG}_heft "$RUNS"/${TAG}_gmhrl \
    --bound --bound_time 3600 --bound_workers 8 --out "$RUNS/summary_${TAG}" ;;
esac
