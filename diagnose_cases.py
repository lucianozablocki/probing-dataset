"""Diagnose every alignment case independently, instead of first-failure-wins.

plot_probing.py reports one reason per pdb_id because each check returns as soon
as it fires, and the checks are ordered: the rnaglib-vs-rnapdbee sequence
mismatch (return 2) is tested before the seqB gap check (return 1). A pdb_id
with both problems is therefore only ever reported as a tool diff, so it never
reaches SEQB_GAPS -- and remove_gap.py skips the tool-diff ids anyway, which
guarantees its gaps survive. The conditions are independent properties, not
ranked failure modes, so this script evaluates all of them for every case.

Source of truth is structure_and_probing.csv, NOT the alignments parquet. The
parquet still holds pre-correction alignments -- the fixes in
rnaglib_rnapdbee_diff/updated_aligments/ were aggregated into tool_mismatch.csv
and merged downstream without the parquet ever being regenerated -- so
diagnosing against it re-reports 15 tool diffs that were resolved long ago.

The csv also carries `chain` per row, so each case is diagnosed per
(pdb_id, chain) rather than against whichever chain happened to sort first.

Nothing here plots, and plot_probing.py is untouched.
"""
import sys

import pandas as pd

# the universe is ANALYZED_PDBIDS, imported rather than copied so the two
# scripts cannot drift apart the way the hardcoded lists already have
sys.path.insert(0, "rnaglib_rnapdbee_diff")
from compare_sequences import ANALYZED_PDBIDS  # noqa: E402

STRUCTURE_AND_PROBING = "structure_and_probing.csv"
RNAPDB_CSV = "rna_pdb_dataset_bp.csv"
OUT_CSV = "pdb_gaps/case_diagnosis.csv"


def find_alignment_bounds(aligned_pdb_seq):
    """Start and end of the pdb sequence in the alignment (first/last non-gap)."""
    start = end = None
    for idx, nuc in enumerate(aligned_pdb_seq):
        if nuc != "-":
            if start is None:
                start = idx
            end = idx
    return start, end


def interior_gap_columns(aligned_pdb_seq):
    """Gap columns that fall strictly inside the pdb sequence's own bounds.

    Only these shift the reactivity frame; leading and trailing gaps are just
    the alignment window. Mirrors the inside_alignment_pos test in remove_gap.py.
    """
    start, end = find_alignment_bounds(aligned_pdb_seq)
    if start is None:
        return []
    return [i for i, c in enumerate(aligned_pdb_seq) if c == "-" and start < i < end]


def diagnose(pdb_id, chain, group, rnapdbee):
    ungapped = group.aligned_pdb_seq.str.replace("-", "", regex=False)
    gaps = group.aligned_pdb_seq.map(interior_gap_columns)

    rec = {
        "pdb_id": pdb_id,
        "chain": chain,
        "n_rows": len(group),
        # the alignment reference disagrees with the structure's own sequence
        "tool_diff": bool((ungapped != group.sequence).any()),
        "n_rows_tool_diff": int((ungapped != group.sequence).sum()),
        # gaps on the pdb side of the alignment, which shift the reactivity frame
        "seqb_gaps": bool(gaps.map(bool).any()),
        "n_rows_with_seqb_gaps": int(gaps.map(bool).sum()),
        "n_gap_cols_inside_bounds": int(gaps.map(len).sum()),
        # what our pipeline actually drops on: the two frames disagree in length
        "frame_mismatch": bool(
            (group.aligned_pdb_seq.str.len() != group.sequence.str.len()).any()
        ),
        "n_rows_frame_mismatch": int(
            (group.aligned_pdb_seq.str.len() != group.sequence.str.len()).sum()
        ),
        "struct_len_mismatch": bool(
            (group.dot_bracket.str.len() != group.sequence.str.len()).any()
        ),
    }

    # has rnapdbee moved since this csv was built?
    ref = rnapdbee.get((pdb_id, chain))
    rec["rnapdbee_drift"] = None if ref is None else bool((group.sequence != ref).any())
    rec["missing_from_rnapdbee"] = ref is None

    rec["reasons"] = ";".join(
        name for name in
        ("tool_diff", "seqb_gaps", "frame_mismatch", "struct_len_mismatch",
         "rnapdbee_drift", "missing_from_rnapdbee")
        if rec[name]
    ) or "ok"
    return rec


def main():
    df = pd.read_csv(
        STRUCTURE_AND_PROBING,
        usecols=["pdb_id", "chain", "experiment", "aligned_pdb_seq",
                 "aligned_rnagym_seq", "sequence", "dot_bracket"],
    )
    pdbee = pd.read_csv(RNAPDB_CSV)
    rnapdbee = {
        (r.id.strip().lower(), str(r.chain).strip()): r.sequence.strip()
        for r in pdbee.itertuples()
    }

    universe = set(ANALYZED_PDBIDS)
    in_universe = df[df.pdb_id.isin(universe)]

    diag = pd.DataFrame(
        diagnose(pdb_id, chain, group, rnapdbee)
        for (pdb_id, chain), group in in_universe.groupby(["pdb_id", "chain"])
    ).sort_values(["pdb_id", "chain"])
    diag.to_csv(OUT_CSV, index=False)

    print(f"{len(diag)} (pdb_id, chain) cases diagnosed over "
          f"{diag.pdb_id.nunique()} pdb_ids (universe: ANALYZED_PDBIDS)")
    outside = sorted(set(df.pdb_id) - universe)
    if outside:
        print(f"in {STRUCTURE_AND_PROBING} but not analyzed: {', '.join(outside)}")
    absent = sorted(universe - set(df.pdb_id))
    if absent:
        print(f"in ANALYZED_PDBIDS but absent from {STRUCTURE_AND_PROBING} "
              f"({len(absent)}): {', '.join(absent)}")
    print()
    print("reason combinations:")
    print(diag.reasons.value_counts().to_string())

    both = diag[diag.tool_diff & diag.seqb_gaps]
    print(f"\n{len(both)} cases have BOTH a tool diff and seqB gaps "
          f"(invisible to plot_probing.py's ordered returns):")
    print("  " + (", ".join(f"{r.pdb_id}_{r.chain}" for r in both.itertuples())
                  if len(both) else "(none)"))

    gaps = sorted(diag[diag.seqb_gaps].pdb_id.unique())
    tooldiff = sorted(diag[diag.tool_diff].pdb_id.unique())
    print(f"\n# every pdb_id with interior gaps on the pdb side ({len(gaps)}), "
          f"regardless of what else is wrong")
    print(f"SEQB_GAPS = {gaps}")
    print(f"\n# every pdb_id whose alignment disagrees with its own structure "
          f"sequence ({len(tooldiff)})")
    print(f"TOOL_DIFF = {tooldiff}")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
