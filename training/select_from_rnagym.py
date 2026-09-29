"""Select sequences from RNAgym's train_data.csv that are not already used in
structure_and_probing.csv, and write them out as id,sequence,reactivity rows.

Selection criteria:
    - sequence_id not present in structure_and_probing.csv's rnagym_id column
    - experiment_type == "2A3_MaP"
    - SN_filter == 1
    - len(sequence) <= 512
The per-position reactivity_XXXX columns are collapsed into a single
`reactivity` column holding a JSON-style list of floats (NaNs -> -1000),
following the convention used in probing_postprocess.py. reactivity_XXXX
always has 206 columns (the Ribonanza layout); the list is trimmed down to
len(sequence) since real sequences here are shorter, with the tail padded
as the -1000 no-data sentinel.

The whole file is scanned and the n selected rows are drawn with reservoir
sampling (Algorithm R), so they are a uniform random sample of every row
that passes the filters, not just the first n found -- unlike taking a fixed
percentage of each chunk, this needs no estimate of the file's overall
selectivity to land on an exact count.
"""

import argparse
import ast
import csv
import random
import sys

import pandas as pd

REACTIVITY_COLS = [f"reactivity_{i:04d}" for i in range(1, 207)]
EXPERIMENT_TYPE = "2A3_MaP"


def load_excluded_ids(structure_and_probing_csv):
    df = pd.read_csv(structure_and_probing_csv, usecols=["rnagym_id"])
    return set(df["rnagym_id"].astype(str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="/home/lzablocki/probing-dataset/train_data.csv")
    parser.add_argument(
        "--structure-and-probing-csv",
        default="/home/lzablocki/probing-dataset/structure_and_probing.csv",
    )
    parser.add_argument("--output", default="/home/lzablocki/probing-dataset/rnagym_selected.csv")
    parser.add_argument("--n", type=int, default=20000, help="number of sequences to extract")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    excluded_ids = load_excluded_ids(args.structure_and_probing_csv)
    print(f"Loaded {len(excluded_ids)} rnagym_id values to exclude", file=sys.stderr)

    usecols = ["sequence_id", "sequence", "experiment_type", "SN_filter"] + REACTIVITY_COLS

    n_total = 0
    n_eligible = 0
    n_dropped_experiment_type = 0
    n_dropped_sn_filter = 0
    n_dropped_length = 0
    n_dropped_excluded_id = 0
    n_dropped_duplicate_id = 0
    seen_ids = set()  # sequence_id can repeat in train_data.csv (replicate reads); keep the first
    reservoir = []  # Algorithm R: uniform sample of size args.n over every eligible row seen so far

    reader = pd.read_csv(args.train_csv, usecols=usecols, chunksize=args.chunksize)
    for chunk_idx, chunk in enumerate(reader):
        print(f"Processing chunk {chunk_idx} ({n_eligible} eligible rows seen so far)", file=sys.stderr)
        n_total += len(chunk)

        before = len(chunk)
        chunk = chunk[chunk["experiment_type"] == EXPERIMENT_TYPE]
        n_dropped_experiment_type += before - len(chunk)

        before = len(chunk)
        chunk = chunk[chunk["SN_filter"] == 1]
        n_dropped_sn_filter += before - len(chunk)

        before = len(chunk)
        chunk = chunk[chunk["sequence"].str.len() <= args.max_len]
        n_dropped_length += before - len(chunk)

        before = len(chunk)
        chunk = chunk[~chunk["sequence_id"].astype(str).isin(excluded_ids)]
        n_dropped_excluded_id += before - len(chunk)

        before = len(chunk)
        chunk = chunk[~chunk["sequence_id"].astype(str).isin(seen_ids)]
        chunk = chunk.drop_duplicates(subset="sequence_id")
        n_dropped_duplicate_id += before - len(chunk)

        if chunk.empty:
            continue

        seen_ids.update(chunk["sequence_id"].astype(str))
        reactivity_values = chunk[REACTIVITY_COLS].fillna(-1000).values.tolist()

        for (_, row), reactivity in zip(chunk.iterrows(), reactivity_values):
            reactivity = reactivity[: len(row["sequence"])]
            item = (row["sequence_id"], row["sequence"], reactivity)

            if len(reservoir) < args.n:
                reservoir.append(item)
            else:
                j = rng.randint(0, n_eligible)
                if j < args.n:
                    reservoir[j] = item
            n_eligible += 1

    with open(args.output, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["id", "sequence", "reactivity"])
        writer.writerows(reservoir)

    print(f"Rows scanned: {n_total}", file=sys.stderr)
    print(f"  dropped (experiment_type != {EXPERIMENT_TYPE}): {n_dropped_experiment_type}", file=sys.stderr)
    print(f"  dropped (SN_filter != 1): {n_dropped_sn_filter}", file=sys.stderr)
    print(f"  dropped (len(sequence) > {args.max_len}): {n_dropped_length}", file=sys.stderr)
    print(f"  dropped (sequence_id in structure_and_probing.csv): {n_dropped_excluded_id}", file=sys.stderr)
    print(f"  dropped (duplicate sequence_id, already seen): {n_dropped_duplicate_id}", file=sys.stderr)
    print(f"Eligible rows: {n_eligible}", file=sys.stderr)
    print(f"Wrote {len(reservoir)} sequences to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
