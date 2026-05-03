#!/usr/bin/env python3
"""
delete_from_probing.py

Takes a pdb_id and a position (0-indexed within seqB) as input.
For each alignment row with that pdb_id, computes the aligned position as:
    aligned_position = input_position + initial_gaps_in_seqB
Then removes that column from alignment_seqA, reactivity, and reactivity_errors.

Output CSV columns: pdb_id, rnagym_id, reactivity, reactivity_errors, alignment_seqA, alignment_seqB.
"""

import argparse
import pandas as pd

PARQUET_URL = (
    "https://raw.githubusercontent.com/lucianozablocki/probing-dataset"
    "/refs/heads/main/rnagym_vs_rnapdb_alignments_postprocessed.parquet"
)


def find_alignment_bounds(alignment_seqB):
    """Find start and end of seqB in the alignment (first/last non-gap positions)."""
    start = None
    end = None
    for idx, nuc in enumerate(alignment_seqB):
        if nuc != '-':
            if start is None:
                start = idx
            end = idx
    return start, end


def main():
    parser = argparse.ArgumentParser(
        description="Delete a position from probing data for a given PDB ID."
    )
    parser.add_argument("pdb_id", type=str, help="PDB ID to filter rows by")
    parser.add_argument(
        "position",
        type=int,
        help="Position within seqB (0-indexed) to remove from each alignment",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: <pdb_id>_pos<position>_deleted.csv)",
    )
    args = parser.parse_args()

    pdb_id = args.pdb_id.lower()
    input_position = args.position
    output_path = args.output or f"{pdb_id}_pos{input_position}_deleted.csv"

    print("Loading parquet file...")
    df = pd.read_parquet(PARQUET_URL)

    seen=[]
    count=0
    rows_for_pdb=[]
    rnagym_seqs_aligned_with_pdb_id = df[df['pdb_id']==pdb_id].reset_index(drop=True)
    for _, row in rnagym_seqs_aligned_with_pdb_id.iterrows():
        if (row['pdb_id'], row['sequence_id'], row['experiment_type']) in seen:
            continue
        seen.append((row['pdb_id'], row['sequence_id'], row['experiment_type']))
        count+=1
        rows_for_pdb.append(row)
    print(f"Found {len(rows_for_pdb)} rows for pdb_id '{pdb_id}'.")

    results = []

    for idx, row in enumerate(rows_for_pdb):
        alignment_seqB = row.get('alignment_seqB')
        if not alignment_seqB:
            print(f"CRITICAL: row {row['sequence_id']} has no alignment_seqB, fix and try again.")
            break

        start, _ = find_alignment_bounds(alignment_seqB)
        if start is None:
            print(f"CRITICAL: row {row['sequence_id']} has all-gap seqB, fix and try again.")
            break

        alignment_seqA = row.get('alignment_seqA', '')

        # Map input_position (0-indexed in original seqB) to the alignment column.
        # We walk alignment_seqB and collect columns where seqB has a real nucleotide;
        # the input_position-th such column is the one to delete.
        seqB_nongap_cols = [i for i, c in enumerate(alignment_seqB) if c != '-']
        if input_position >= len(seqB_nongap_cols):
            print(
                f"CRITICAL: input_position {input_position} out of range for "
                f"row {row['sequence_id']} (seqB has {len(seqB_nongap_cols)} nucleotides), fix and try again."
            )
            break
        aligned_position = seqB_nongap_cols[input_position]

        reactivity = list(row.get('reactivity', []))
        reactivity_errors = list(row.get('reactivity_errors', []))

        results.append({
            'pdb_id': row['pdb_id'],
            'rnagym_id': row['sequence_id'],
            'reactivity': reactivity[:aligned_position] + reactivity[aligned_position + 1:],
            'reactivity_errors': reactivity_errors[:aligned_position] + reactivity_errors[aligned_position + 1:],
            'alignment_seqA': alignment_seqA[:aligned_position] + alignment_seqA[aligned_position + 1:],
            'alignment_seqB': alignment_seqB[:aligned_position] + alignment_seqB[aligned_position + 1:],
        })

    if not results:
        print("No valid rows to output.")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Written {len(results)} rows to '{output_path}'.")


if __name__ == "__main__":
    main()
