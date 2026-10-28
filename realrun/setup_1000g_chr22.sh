#!/usr/bin/env bash
# Prepare the real-data experiment (plan B): 50 samples of the 1000 Genomes
# high-coverage (30x, GRCh38) data set, restricted to reads mapped to chr22.
# Steps: 1) install tools with micromamba, 2) build a chr22 reference,
# 3) choose 50 samples (fixed random seed), 4) download the chr22 slice of each
# sample's CRAM and convert it to paired FASTQ.
# Every step is restartable; finished samples are skipped.
#
# Usage:  W=/mnt/e/ForPhD/Data/wgs_chr22 REF_FULL=/path/GRCh38_full_analysis_set_plus_decoy_hla.fa \
#         bash realrun/setup_1000g_chr22.sh [tools] [ref] [samples] [fetch]
set -euo pipefail
W=${W:-/mnt/e/ForPhD/Data/wgs_chr22}
REF_FULL=${REF_FULL:-/mnt/e/ForPhD/Code/创新点3/old_version/1000genome/GRCh38_full_analysis_set_plus_decoy_hla.fa}
NS=${NS:-50}
SEED=${SEED:-2026}
FETCH_JOBS=${FETCH_JOBS:-3}
INDEX_URL=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/1000G_2504_high_coverage.sequence.index
mkdir -p "$W"/{bin,ref,fastq,slices,logs}
export MAMBA_ROOT_PREFIX="$W/mamba"
ENV="$W/env"
run() { "$W/bin/micromamba" run -p "$ENV" "$@"; }

step_tools() {
  if [ ! -x "$W/bin/micromamba" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$W/bin" --strip-components=1 bin/micromamba
  fi
  "$W/bin/micromamba" create -y --always-copy -p "$ENV" -c conda-forge -c bioconda \
      "samtools>=1.19" "bcftools>=1.19" "htslib>=1.19" bwa fastqc gatk4 freebayes multiqc python=3.11
  run samtools --version | head -1; run bwa 2>&1 | grep Version; run gatk --version | head -2
  run freebayes --version; run fastqc --version
}

step_ref() {
  [ -f "$REF_FULL" ] || { echo "reference not found: $REF_FULL"; exit 1; }
  [ -f "$REF_FULL.fai" ] || run samtools faidx "$REF_FULL"
  cd "$W/ref"
  [ -f chr22.fa ] || run samtools faidx "$REF_FULL" chr22 > chr22.fa
  [ -f chr22.fa.fai ] || run samtools faidx chr22.fa
  [ -f chr22.dict ] || run samtools dict chr22.fa -o chr22.dict
  [ -f chr22.fa.bwt ] || run bwa index chr22.fa
  ls -la "$W/ref"
}

step_samples() {
  cd "$W"
  if [ ! -s sequence.index ]; then
    curl -fL --retry 5 -o sequence.index "$INDEX_URL" || {
      echo "Could not download $INDEX_URL"
      echo "Open https://www.internationalgenome.org/data-portal/data-collection/30x-grch38 ,"
      echo "download the sequence index of the 2504-sample collection and save it as $W/sequence.index"; exit 1; }
  fi
  python3 - "$NS" "$SEED" <<'EOF'
import sys, random, csv
ns, seed = int(sys.argv[1]), int(sys.argv[2])
rows, header = [], None
for line in open("sequence.index", encoding="utf-8"):
    if line.startswith("##"):
        continue
    f = line.rstrip("\n").split("\t")
    if header is None and line.startswith("#"):
        header = [h.lstrip("#") for h in f]
        continue
    if header is None or len(f) < len(header):
        continue
    r = dict(zip(header, f))
    url = r.get("ENA_FILE_PATH", "")
    if url.endswith(".cram"):
        rows.append((r.get("SAMPLE_NAME") or url.rsplit("/", 1)[1].split(".")[0],
                     url.replace("ftp://", "https://")))
rows = sorted(set(rows))
print(f"{len(rows)} high-coverage CRAMs in the index")
pick = sorted(random.Random(seed).sample(rows, ns))
with open("samples.tsv", "w") as fh:
    fh.write("sample\tcram_url\n")
    for s, u in pick:
        fh.write(f"{s}\t{u}\n")
print(f"wrote samples.tsv with {len(pick)} samples (random seed {seed})")
EOF
  head -3 samples.tsv
}

fetch_one() {  # sample url
  local s=$1 u=$2 out="$W/fastq/${s}_R1.fastq.gz"
  [ -s "$out" ] && [ -s "$W/fastq/${s}_R2.fastq.gz" ] && return 0
  for attempt in 1 2 3 4 5 6; do
    local src=$u; [ $((attempt % 2)) -eq 0 ] && src=${u/https:\/\//ftp://}   # alternate https / ftp
    if run samtools view -@2 -T "$REF_FULL" -b -o "$W/slices/$s.chr22.bam" "$src" chr22 2>>"$W/logs/fetch_$s.log"; then
      run samtools collate -@2 -O -u "$W/slices/$s.chr22.bam" "$W/slices/$s.tmp" \
        | run samtools fastq -@2 -n -1 "$W/fastq/${s}_R1.fastq.gz" -2 "$W/fastq/${s}_R2.fastq.gz" \
              -0 /dev/null -s /dev/null - 2>>"$W/logs/fetch_$s.log" && {
        rm -f "$W/slices/$s.chr22.bam"; echo "ok $s"; return 0; }
    fi
    echo "retry $s ($attempt)" >> "$W/logs/fetch_$s.log"; sleep 30
  done
  echo "FAILED $s"; return 1
}
export -f fetch_one run; export W REF_FULL ENV MAMBA_ROOT_PREFIX

step_fetch() {
  cd "$W"
  tail -n +2 samples.tsv | xargs -P "$FETCH_JOBS" -n 2 bash -c 'fetch_one "$0" "$1"' || true
  n=$(ls "$W"/fastq/*_R2.fastq.gz 2>/dev/null | wc -l)
  echo "samples with FASTQ: $n / $NS"; du -sh "$W/fastq"
}

for p in ${*:-tools ref samples fetch}; do echo "=== $p ($(date)) ==="; step_$p; done
