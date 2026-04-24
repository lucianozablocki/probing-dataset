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
        print(count)
        print(row['sequence_id'])
        print(row['experiment_type'])
        rows_for_pdb.append(row)
    print(f"Found {len(rows_for_pdb)} rows for pdb_id '{pdb_id}'.")

    results = []
    skipped = 0

    for idx, row in enumerate(rows_for_pdb):
        alignment_seqB = row.get('alignment_seqB')
        if not alignment_seqB:
            print(f"Warning: row {row['sequence_id']} has no alignment_seqB, skipping.")
            skipped += 1
            continue

        start, _ = find_alignment_bounds(alignment_seqB)
        if start is None:
            print(f"Warning: row {row['sequence_id']} has all-gap seqB, skipping.")
            skipped += 1
            continue

        alignment_seqA = row.get('alignment_seqA', '')
        # if idx==14:
        # print(alignment_seqA)
        # print(row.get('alignment_seqB'))

        # Compute aligned position: start of seqB + input_position + gaps in alignment_seqA
        # within the [start, start+input_position) region.
        gaps_in_region = alignment_seqA[start:start + input_position].count('-')
        aligned_position = start + input_position + gaps_in_region

        if aligned_position >= len(alignment_seqA):
            print(
                f"Warning: aligned_position {aligned_position} out of bounds for "
                f"row {row['sequence_id']} (alignment len={len(alignment_seqA)}), skipping."
            )
            skipped += 1
            continue

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
    print(f"Written {len(results)} rows to '{output_path}' (skipped {skipped}).")


if __name__ == "__main__":
    main()
