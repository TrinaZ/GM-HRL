#!/usr/bin/env bash
# Pilot on the first K samples: run every stage once, sequentially, and record
# wall time and peak memory. Used ONLY to set est_seconds / resource requests in
# the pipeline specification and to choose the branch statistic and threshold.
# Pilot samples are excluded from the 50-sample experiment.
# Usage: W=/mnt/e/ForPhD/Data/wgs_chr22 K=5 bash realrun/wgs_chr22/pilot.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-/mnt/e/ForPhD/Data/wgs_chr22}; K=${K:-5}
export W OUT=${OUT:-$W/pilot}
mkdir -p "$OUT"
LOG=$OUT/pilot_times.tsv
[ -f "$LOG" ] || printf "sample\tstage\tthreads\tseconds\tmax_rss_mb\texit\n" > "$LOG"
declare -A TH=([fastqc]=2 [bwa]=8 [sort]=4 [index]=2 [gatk]=4 [freebayes]=1 [bcftools]=1 [multiqc]=1)
mapfile -t SAMPLES < <(tail -n +2 "$W/pilot_samples.tsv" | cut -f1 | head -n "$K")
for s in "${SAMPLES[@]}"; do
  for st in fastqc bwa sort index gatk freebayes bcftools multiqc; do
    awk -F'	' -v s="$s" -v st="$st" '$1==s && $2==st && $6==0{f=1} END{exit !f}' "$LOG" && continue
    t0=$EPOCHREALTIME; echo 0 > "$OUT/.rss"
    if [ -x /usr/bin/time ]; then
      /usr/bin/time -f "%M" -o "$OUT/.rss" bash "$HERE/stage.sh" "$st" "$s" "${TH[$st]}" > "$OUT/$s.$st.log" 2>&1; rc=$?
    else
      bash "$HERE/stage.sh" "$st" "$s" "${TH[$st]}" > "$OUT/$s.$st.log" 2>&1; rc=$?
    fi
    t1=$EPOCHREALTIME
    rss=$(tail -1 "$OUT/.rss" 2>/dev/null); [[ "$rss" =~ ^[0-9]+$ ]] || rss=0
    awk -v s="$s" -v st="$st" -v th="${TH[$st]}" -v a="$t0" -v b="$t1" -v r="$rss" -v rc="$rc" \
      'BEGIN{printf "%s\t%s\t%s\t%.1f\t%.0f\t%s\n", s, st, th, b-a, r/1024, rc}' | tee -a "$LOG"
    [ $rc -eq 0 ] || { echo "stage $st failed for $s, see $OUT/$s.$st.log"; break; }
  done
done
echo; echo "QC statistics (candidate branch metrics):"
for s in "${SAMPLES[@]}"; do [ -f "$OUT/$s/$s.qc.json" ] && echo "$s $(tr -d '\n ' < "$OUT/$s/$s.qc.json")"; done
