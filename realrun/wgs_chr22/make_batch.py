#!/usr/bin/env python3
"""Write the batch file (sample <TAB> arrival offset in s) for the real run.
Arrivals: Poisson process with mean gap GAP s (fixed seed), the same file is
used for every scheduler. Pilot samples are excluded.
Usage: python3 realrun/wgs_chr22/make_batch.py W GAP SEED OUT"""
import random, sys
from pathlib import Path
W, gap, seed, out = Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
read = lambda p: [l.split("\t")[0] for l in open(p).read().splitlines()[1:] if l.strip()]
samples, pilot = read(W / "samples.tsv"), set(read(W / "pilot_samples.tsv"))
assert not (set(samples) & pilot), f"pilot samples in batch: {set(samples) & pilot}"
missing = [s for s in samples if not (W / "fastq" / f"{s}_R1.fastq.gz").exists()
           or not (W / "fastq" / f"{s}_R2.fastq.gz").exists()]
assert not missing, f"FASTQ missing for {missing}"
rng, t = random.Random(seed), 0.0
with open(out, "w") as f:
    for k, s in enumerate(samples):
        f.write(f"{s}\t{t:.0f}\n")
        t += rng.expovariate(1.0 / gap)
print(f"{len(samples)} samples, last arrival at {t/3600:.2f} h -> {out}")
