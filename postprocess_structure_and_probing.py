import json
from ast import literal_eval

import pandas as pd


INPUT_CSV = "structure_and_probing_raw.csv"
OUTPUT_CSV = "structure_and_probing.csv"

# consolidate_csvs.py carries the structure column through from
# rna_pdb_dataset_bp.csv under its name there; downstream consumers all read
# dot_bracket, so rename it here rather than by hand after the fact
RENAME_COLUMNS = {"base_pairs": "dot_bracket"}

LIST_COLUMNS_TO_CROP = ["reactivity", "reactivity_errors"]
SEQ_COLUMNS_TO_CROP = ["aligned_pdb_seq", "aligned_rnagym_seq"]


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

    # Crop only the desired list columns.
    for col in LIST_COLUMNS_TO_CROP:
        parsed = literal_eval(row[col])
        df.at[idx, col] = parsed[start : end + 1]

    # Crop alignment columns as strings.
    for col in SEQ_COLUMNS_TO_CROP:
        df.at[idx, col] = row[col][start : end + 1]

# Serialize list columns right before writing.
for col in LIST_COLUMNS_TO_CROP:
    df[col] = df[col].apply(json.dumps)

df = df.rename(columns=RENAME_COLUMNS)
assert "dot_bracket" in df.columns, "structure column is missing"

df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote cropped file: {OUTPUT_CSV}")
