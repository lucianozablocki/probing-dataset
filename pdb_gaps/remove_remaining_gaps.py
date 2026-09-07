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

Reactivity is indexed by alignment column here -- postprocess_structure_and_probing.py
crops it with alignment bounds -- so removing column i removes reactivity[i].
The arrays run past the end of the alignment (the rest of the rnagym read), and
that tail is left alone.
"""
import json
from ast import literal_eval

import pandas as pd

DIAGNOSIS = "pdb_gaps/case_diagnosis.csv"
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
    """Drop the interior gap columns from both alignments and both list columns."""
    cols = interior_gap_columns(row["aligned_pdb_seq"])
    if not cols:
        return None

    seqs = {c: row[c] for c in SEQ_COLUMNS}
    lists = {c: literal_eval(row[c]) for c in LIST_COLUMNS}
    for col in sorted(cols, reverse=True):
        for c in SEQ_COLUMNS:
            seqs[c] = seqs[c][:col] + seqs[c][col + 1:]
        for c in LIST_COLUMNS:
            lists[c] = lists[c][:col] + lists[c][col + 1:]

    out = row.to_dict()
    out.update(seqs)
    out.update({c: json.dumps(v) for c, v in lists.items()})
    return out


def main():
    diag = pd.read_csv(DIAGNOSIS)
    cases = set(map(tuple, diag[diag.seqb_gaps][["pdb_id", "chain"]].values))
    print(f"{len(cases)} (pdb_id, chain) cases with interior gaps, "
          f"{len({p for p, _ in cases})} pdb_ids")

    results, seen_source = [], {}
    for path in SOURCES:
        df = pd.read_csv(path)
        hit = df[df[["pdb_id", "chain"]].apply(tuple, axis=1).isin(cases)]
        n = 0
        for _, row in hit.iterrows():
            fixed = degap(row)
            if fixed is not None:
                results.append(fixed)
                n += 1
        seen_source[path] = n
        print(f"  {path}: {len(hit)} rows in scope, {n} carried gaps")

    out = pd.DataFrame(results)
    assert not out.empty, "no rows to fix"
    assert not out.duplicated(subset=KEY).any(), "a row was fixed from two sources"

    # postconditions: the frame is repaired and nothing else moved
    ungapped = out.aligned_pdb_seq.str.replace("-", "", regex=False)
    assert (ungapped == out.sequence).all(), "degapped alignment != structure sequence"
    assert not out.aligned_pdb_seq.map(interior_gap_columns).map(bool).any(), \
        "interior gaps remain"
    lengths = out.reactivity.map(lambda v: len(literal_eval(v)))
    assert (lengths >= out.aligned_pdb_seq.str.len()).all(), \
        "reactivity is shorter than its alignment"

    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows to {OUT_CSV}")
    print(out.groupby("pdb_id").size().to_string())


if __name__ == "__main__":
    main()
