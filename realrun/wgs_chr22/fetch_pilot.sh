#!/usr/bin/env bash
# Choose K pilot samples that are NOT among the 50 experiment samples and
# download their chr22 reads (same procedure as setup_1000g_chr22.sh).
# Run after setup_1000g_chr22.sh has finished its 'samples' step.
set -euo pipefail
W=${W:-/mnt/e/ForPhD/Data/wgs_chr22}
REF_FULL=${REF_FULL:-/mnt/e/ForPhD/Code/创新点3/old_version/1000genome/GRCh38_full_analysis_set_plus_decoy_hla.fa}
K=${K:-5}
export MAMBA_ROOT_PREFIX="$W/mamba"
run() { "$W/bin/micromamba" run -p "$W/env" "$@"; }
cd "$W"
python3 - "$K" <<'PY'
import sys, random
k = int(sys.argv[1])
used = {l.split("\t")[0] for l in open("samples.tsv").read().splitlines()[1:]}
rows, header = [], None
for line in open("sequence.index", encoding="utf-8"):
    if line.startswith("##"):
        continue
    f = line.rstrip("\n").split("\t")
    if header is None and line.startswith("#"):
        header = [h.lstrip("#") for h in f]; continue
    if header is None or len(f) < len(header):
        continue
    r = dict(zip(header, f)); url = r.get("ENA_FILE_PATH", "")
    if url.endswith(".cram"):
        s = r.get("SAMPLE_NAME") or url.rsplit("/", 1)[1].split(".")[0]
        if s not in used:
            rows.append((s, url.replace("ftp://", "https://")))
pick = sorted(random.Random(7).sample(sorted(set(rows)), k))
with open("pilot_samples.tsv", "w") as fh:
    fh.write("sample\tcram_url\n")
    for s, u in pick: fh.write(f"{s}\t{u}\n")
print("pilot samples:", [s for s, _ in pick])
PY
mkdir -p fastq slices logs
tail -n +2 pilot_samples.tsv | while IFS=$'\t' read -r s u; do
  [ -s "fastq/${s}_R2.fastq.gz" ] && { echo "have $s"; continue; }
  ok=0
  for attempt in 1 2 3 4 5 6; do
    if run samtools view -@2 -T "$REF_FULL" -b -o "slices/$s.chr22.bam" "$u" chr22 2>>"logs/fetch_$s.log"; then
      run samtools collate -@2 -O -u "slices/$s.chr22.bam" "slices/$s.tmp" \
        | run samtools fastq -@2 -n -1 "fastq/${s}_R1.fastq.gz" -2 "fastq/${s}_R2.fastq.gz" -0 /dev/null -s /dev/null - \
        2>>"logs/fetch_$s.log" && { rm -f "slices/$s.chr22.bam"; echo "ok $s"; ok=1; break; }
    fi
    echo "retry $s ($attempt)" >> "logs/fetch_$s.log"; sleep 30
  done
  [ $ok -eq 1 ] || echo "FAILED $s"
done
