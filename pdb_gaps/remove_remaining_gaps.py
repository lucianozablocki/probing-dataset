"""Remove interior pdb-side alignment gaps from the cases remove_gap.py missed.

remove_gap.py only ever iterated a hardcoded SEQB_GAPS list and skipped the
pdb_ids whose alignments had been regenerated, so 18 pdb_ids / 931 rows kept
their gaps -- see diagnose_cases.py. Those gaps shift reactivity relative to the
pdb sequence, which is why lp_eval drops the rows outright.

Input is the two consolidate_csvs.py sources the affected rows actually live in,
NOT rnagym_vs_rnapdb_alignments_postprocessed.parquet. The parquet still holds
pre-correction alignments for 7 of these pdb_ids (1e8o, 3k1v, 5d5l, 6prv, 6xko,
7mky, 8g9z), so sourcing from it would silently undo the tool-diff fixes in
rnaglib_rnapdbee_diff/updated_aligments/.

Output is a fourth csv for consolidate_csvs.py to merge, which supersedes the
original rows by key. Nothing existing is edited: remove_gap.py and its
alignments_seqb_gaps_removed.csv (a disjoint set of 28 pdb_ids) stay as they are.

Reactivity has one slot per rnagym NUCLEOTIDE, not per alignment column: it is
copied verbatim from the parquet by complete_rows.py and aggregate_csv.py, and
never re-indexed. So erasing alignment column i erases the reactivity value at
i minus the number of rnagym gaps before it -- the same arithmetic remove_gap.py
uses, kept identical here so there is one definition of the operation.

Confirmed empirically: on the DMS rows where the two readings differ, grouping
reactivity per nucleotide separates A/C from G/U by 7.8x (control: 5.1x), while
grouping per column gives 1.6x -- the smear of an off-by-n frame shift.

A column where the rnagym row is itself a gap has no reactivity value to erase.
The arrays run past the end of the alignment (the rest of the rnagym read), and
that tail is left alone.
"""
import json
from ast import literal_eval

import pandas as pd

SOURCES = [
    "rnaglib_rnapdbee_diff/tool_mismatch.csv",
    "no_transformations/alignments_rows_completed.csv",
]
OUT_CSV = "pdb_gaps/alignments_remaining_gaps_removed.csv"

KEY = ["pdb_id", "rnagym_id", "experiment", "chain"]
LIST_COLUMNS = ["reactivity", "reactivity_errors"]
SEQ_COLUMNS = ["aligned_pdb_seq", "aligned_rnagym_seq"]


def find_alignment_bounds(alignment_seq):
    """Return first/last non-gap indices (inclusive) for an alignment string."""
    start = end = None
    for idx, nuc in enumerate(alignment_seq):
        if nuc != "-":
            if start is None:
                start = idx
            end = idx
    return start, end


def interior_gap_columns(aligned_pdb_seq):
    """Gap columns strictly inside the pdb sequence's own bounds.

    Leading and trailing gaps are just the alignment window and are cropped
    later; only these interior ones shift the frame.
    """
    start, end = find_alignment_bounds(aligned_pdb_seq)
    if start is None:
        return []
    return [i for i, c in enumerate(aligned_pdb_seq) if c == "-" and start < i < end]


def degap(row):
    """Erase each interior gap column from the alignments and the reactivity.

    Columns are removed back to front so earlier deletions do not shift the
    indices still to be processed.
    """
    cols = interior_gap_columns(row["aligned_pdb_seq"])
    if not cols:
        return None

    pdb_seq = row["aligned_pdb_seq"]
    rnagym_seq = row["aligned_rnagym_seq"]
    lists = {c: literal_eval(row[c]) for c in LIST_COLUMNS}

    for col in sorted(cols, reverse=True):
        if rnagym_seq[col] != "-":
            # translate the column into an index of the ungapped rnagym sequence
            i = col - rnagym_seq[:col].count("-")
            for c in LIST_COLUMNS:
                lists[c] = lists[c][:i] + lists[c][i + 1:]
        rnagym_seq = rnagym_seq[:col] + rnagym_seq[col + 1:]
        pdb_seq = pdb_seq[:col] + pdb_seq[col + 1:]

    out = row.to_dict()
    out["aligned_pdb_seq"] = pdb_seq
    out["aligned_rnagym_seq"] = rnagym_seq
    out.update({c: json.dumps(v) for c, v in lists.items()})
    return out


def main():
    """Fix every row in the source csvs that still carries an interior pdb gap.

    Scanning the sources rather than a diagnosis of the final csv keeps this
    self-contained and idempotent: once the rows are fixed there is nothing to
    find, and a newly introduced gap is picked up without updating a list.
    """
    results = []
    for path in SOURCES:
        df = pd.read_csv(path)
        n = 0
        for _, row in df.iterrows():
            fixed = degap(row)
            if fixed is not None:
                results.append(fixed)
                n += 1
        print(f"  {path}: {len(df)} rows, {n} carried interior pdb gaps")

    if not results:
        print("\nno interior pdb gaps left in the sources; nothing to write")
        return

    out = pd.DataFrame(results)
    assert not out.duplicated(subset=KEY).any(), "a row was fixed from two sources"

    # postconditions: the frame is repaired and nothing else moved
    ungapped = out.aligned_pdb_seq.str.replace("-", "", regex=False)
    assert (ungapped == out.sequence).all(), "degapped alignment != structure sequence"
    assert not out.aligned_pdb_seq.map(interior_gap_columns).map(bool).any(), \
        "interior gaps remain"

    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows to {OUT_CSV}")
    print(out.groupby("pdb_id").size().to_string())


if __name__ == "__main__":
    main()
