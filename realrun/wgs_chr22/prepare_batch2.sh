#!/usr/bin/env bash
# Second real-pipeline batch (REALRUN2_SPEC.md): 50 NEW samples of the same
# 1000 Genomes collection, drawn with seed 2027 after excluding the 50 samples
# of the first batch and the 5 pilot samples. Tools, chr22 reference and
# environment of the first batch are reused (symlinks); nothing is re-tuned.
# Usage: W1=/mnt/e/ForPhD/Data/wgs_chr22 W2=/mnt/e/ForPhD/Data/wgs_chr22_b2 \
#        REF_FULL=... bash realrun/wgs_chr22/prepare_batch2.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W1=${W1:-/mnt/e/ForPhD/Data/wgs_chr22}
W2=${W2:-/mnt/e/ForPhD/Data/wgs_chr22_b2}
mkdir -p "$W2"
for d in bin env mamba ref; do [ -e "$W2/$d" ] || ln -s "$W1/$d" "$W2/$d"; done
cp -n "$W1/pilot_samples.tsv" "$W2/pilot_samples.tsv"
cp -n "$W1/sequence.index" "$W2/sequence.index" 2>/dev/null || true
W=$W2 SEED=2027 NS=50 EXCLUDE="$W1/samples.tsv $W1/pilot_samples.tsv" \
  bash "$REPO/realrun/setup_1000g_chr22.sh" samples fetch
python3 - "$W1/samples.tsv" "$W1/pilot_samples.tsv" "$W2/samples.tsv" <<'PY'
import sys
r = lambda p: {l.split("\t")[0] for l in open(p).read().splitlines()[1:] if l.strip()}
old, pilot, new = r(sys.argv[1]), r(sys.argv[2]), r(sys.argv[3])
assert len(new) == 50 and not (new & old) and not (new & pilot), "overlap with earlier samples"
print("batch 2: 50 new samples, no overlap with batch 1 or the pilot")
PY
