"""
Run POLEGON on a SINGER sample and compute a 'expected' ARG.
"""

import os
import pickle
import subprocess
import tszip, tskit
from datetime import datetime
# --- lib --- #

def tag():
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"

# --- implm --- #
infile = snakemake.input.tree
outfile = snakemake.output[0]
with open(snakemake.log.out, "w") as out, open(snakemake.log.err, "w") as err:
    # First, we have to uncompress the SINGER
    ts = tszip.decompress(infile)
    # For now, this code assumes it's using the `shadow` rule
    base = os.path.splitext(infile)[0]
    temp_file = f"{base}.trees"
    ts.dump(temp_file)
    # Second, we have to create a mutation rate map
    ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
    with open(f"{base}.ratemap", 'w') as f:
        positions = ratemap.position
        rates = ratemap.rate
        for i, pos in enumerate(positions):
            if i < len(rates):
                f.write(f"{pos}\t{rates[i]}\n")
            else:
                f.write(f"{pos}\t{rates[-1]}\n")

    invocation = [
        f"{snakemake.params.polegon_binary}",
        "-input", f"{base}.trees",
        "-output", f"{base}.polegon.trees",
        "-map", f"{base}.ratemap",
        "-num_samples", snakemake.params.get("num_samples", 100),
        "-thinning", snakemake.params.get("thinning", 10),
        "-scaling_rep", snakemake.params.get("scaling_rep", 5)
    ]
    print(f"{tag()}", " ".join(invocation), file=out, flush=True)
    process = subprocess.run(invocation, check=False, stdout=out, stderr=err)
    if process.returncode != 0:
        print(f"{tag()} SINGER run failed ({process.returncode})", file=out, flush=True)
    print(f"{tag()} SINGER run ended ({process.returncode})", file=out, flush=True)
    # Third, we have to compress the SINGER
    out_ts = tskit.load(f"{base}.polegon.trees")
    tszip.compress(out_ts, outfile)
