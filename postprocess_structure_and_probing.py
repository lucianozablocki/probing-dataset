import json
from ast import literal_eval

import pandas as pd


INPUT_CSV = "structure_and_probing_raw.csv"
OUTPUT_CSV = "structure_and_probing.csv"

# consolidate_csvs.py carries the structure column through from
# rna_pdb_dataset_bp.csv under its name there; downstream consumers all read
# dot_bracket, so rename it here rather than by hand after the fact
RENAME_COLUMNS = {"base_pairs": "dot_bracket"}

MISSING = -1000  # the sentinel already used throughout for "not measured"

LIST_COLUMNS_TO_CROP = ["reactivity", "reactivity_errors"]
SEQ_COLUMNS_TO_CROP = ["aligned_pdb_seq", "aligned_rnagym_seq"]


def insert_rnagym_gap_sentinels(aligned_rnagym_seq, values):
    """Give the list one entry per alignment column instead of per nucleotide.

    reactivity comes out of the parquet with a slot per rnagym NUCLEOTIDE, so a
    gap column in aligned_rnagym_seq has no entry and every value after it sits
    one place early. The crop below indexes by alignment column, so the two
    disagree wherever the rnagym row has an interior gap.

    A rnagym gap is not like a pdb gap: the structure does have a residue at
    that column, it just was not probed. So the column stays and the missing
    measurement is marked with the usual sentinel rather than dropped.

    Anything past the end of the alignment (the rest of the rnagym read) is
    carried through untouched; the crop discards it either way.
    """
    out, i = [], 0
    for nuc in aligned_rnagym_seq:
        if nuc == "-":
            out.append(MISSING)
        else:
            out.append(values[i] if i < len(values) else MISSING)
            i += 1
    return out + list(values[i:])


def find_alignment_bounds(alignment_seq):
    """Return first/last non-gap indices (inclusive) for an alignment string."""
    start = None
    end = None
    for idx, nuc in enumerate(alignment_seq):
        if nuc != "-":
            if start is None:
                start = idx
            end = idx
    return start, end


df = pd.read_csv(INPUT_CSV)

for col in LIST_COLUMNS_TO_CROP:
    df[col] = df[col].astype("object")

for idx, row in df.iterrows():
    start, end = find_alignment_bounds(row["aligned_pdb_seq"])

    # Crop only the desired list columns. Realign them to alignment columns
    # first, or the slice below is off by the number of preceding rnagym gaps.
    for col in LIST_COLUMNS_TO_CROP:
        parsed = insert_rnagym_gap_sentinels(
            row["aligned_rnagym_seq"], literal_eval(row[col])
        )
        df.at[idx, col] = parsed[start : end + 1]

    # Crop alignment columns as strings.
    for col in SEQ_COLUMNS_TO_CROP:
        df.at[idx, col] = row[col][start : end + 1]

assert (df.reactivity.map(len) == df.aligned_pdb_seq.str.len()).all(), \
    "reactivity does not have one value per alignment column"

# Serialize list columns right before writing.
for col in LIST_COLUMNS_TO_CROP:
    df[col] = df[col].apply(json.dumps)

df = df.rename(columns=RENAME_COLUMNS)
assert "dot_bracket" in df.columns, "structure column is missing"

df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote cropped file: {OUTPUT_CSV}")
