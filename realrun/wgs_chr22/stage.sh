#!/usr/bin/env bash
# One stage of the chr22 WGS pipeline used in the real-pipeline experiment.
# Usage: stage.sh <stage> <sample> <threads>
#   stages: fastqc bwa sort index gatk freebayes bcftools multiqc metric
# Paths (override with environment variables):
#   W     data root prepared by setup_1000g_chr22.sh (fastq/, ref/, env/)
#   OUT   output root; each sample writes to $OUT/<sample>/
#   METRIC  QC statistic printed by the 'metric' stage (default: dup_rate)
set -euo pipefail
STAGE=$1; S=$2; T=${3:-1}
W=${W:-/mnt/e/ForPhD/Data/wgs_chr22}
OUT=${OUT:-$W/work}
METRIC=${METRIC:-dup_rate}
[ -d "$W/env/bin" ] && export PATH="$W/env/bin:$PATH"
REF=$W/ref/chr22.fa
R1=$W/fastq/${S}_R1.fastq.gz; R2=$W/fastq/${S}_R2.fastq.gz
D=$OUT/$S; mkdir -p "$D"; cd "$D"

case $STAGE in
fastqc)
  mkdir -p qc; fastqc -q -t "$T" -o qc "$R1" "$R2" ;;
bwa)
  bwa mem -t "$T" -R "@RG\tID:${S}\tSM:${S}\tPL:ILLUMINA" "$REF" "$R1" "$R2" 2> bwa.log \
    | samtools view -b -o "$S.raw.bam" - ;;
sort)   # name sort -> fixmate -> coordinate sort -> mark duplicates
  samtools sort -n -@ "$T" -m 1G -T tmp_n -o "$S.name.bam" "$S.raw.bam"
  samtools fixmate -m -@ "$T" "$S.name.bam" "$S.fix.bam"
  samtools sort -@ "$T" -m 1G -T tmp_c -o "$S.pos.bam" "$S.fix.bam"
  samtools markdup -@ "$T" -s -f "$S.markdup.txt" "$S.pos.bam" "$S.dedup.bam"
  rm -f "$S.raw.bam" "$S.name.bam" "$S.fix.bam" "$S.pos.bam" ;;
index)  # index + QC statistics (the branch point)
  samtools index -@ "$T" "$S.dedup.bam"
  samtools stats -@ "$T" "$S.dedup.bam" > "$S.stats.txt"
  python3 - "$S" <<'PY'
import json, re, sys
s = sys.argv[1]
sn = {}
for line in open(f"{s}.stats.txt"):
    if line.startswith("SN\t"):
        k, v = line.split("\t")[1:3]
        sn[k.rstrip(":")] = float(v)
md = open(f"{s}.markdup.txt").read()
def grab(key):
    m = re.search(rf"^{key}:\s*(\d+)", md, re.M)
    return float(m.group(1)) if m else float("nan")
reads = sn.get("raw total sequences", float("nan"))
qc = {
    "reads": reads,
    "mapped_rate": sn.get("reads mapped", 0) / reads,
    "properly_paired_rate": sn.get("reads properly paired", 0) / reads,
    "dup_rate": grab("DUPLICATE TOTAL") / grab("READ"),
    "error_rate": sn.get("error rate", float("nan")),
    "mean_insert": sn.get("insert size average", float("nan")),
    "mean_depth": sn.get("bases mapped (cigar)", 0) / 50818468.0,   # chr22 length (GRCh38)
}
json.dump(qc, open(f"{s}.qc.json", "w"), indent=1)
print(json.dumps(qc))
PY
  ;;
metric) # prints one number for the branch rule
  python3 -c "import json;print(json.load(open('$S.qc.json'))['$METRIC'])" ;;
gatk)
  gatk --java-options "-Xmx4g" HaplotypeCaller -R "$REF" -I "$S.dedup.bam" -O "$S.gatk.vcf.gz" \
       --native-pair-hmm-threads "$T" > gatk.log 2>&1 ;;
freebayes)
  freebayes -f "$REF" "$S.dedup.bam" | bgzip -c > "$S.freebayes.vcf.gz"
  tabix -f -p vcf "$S.freebayes.vcf.gz" ;;
bcftools)   # normalise every caller output that exists; keep the union, report statistics
  vcfs=()
  for c in gatk freebayes; do
    if [ -s "$S.$c.vcf.gz" ]; then
      bcftools norm -f "$REF" -m -any -Oz -o "$S.$c.norm.vcf.gz" "$S.$c.vcf.gz"
      bcftools index -f "$S.$c.norm.vcf.gz"; vcfs+=("$S.$c.norm.vcf.gz")
    fi
  done
  [ ${#vcfs[@]} -gt 0 ] || { echo "no caller output"; exit 1; }
  if [ ${#vcfs[@]} -gt 1 ]; then
    bcftools concat -a -D -Oz -o "$S.final.vcf.gz" "${vcfs[@]}"
  else
    cp "${vcfs[0]}" "$S.final.vcf.gz"
  fi
  bcftools index -f "$S.final.vcf.gz"
  bcftools stats "$S.final.vcf.gz" > "$S.vcfstats.txt" ;;
multiqc)
  multiqc -q -f -o multiqc . ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
