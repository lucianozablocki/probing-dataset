"""
Compare sequences between sanitized_rnapdbdataset.csv (rnaglib) and rna_pdb_dataset_bp.csv (rnapdbee).

For each (pdbid, chain) in the rnaglib file (which has one representative chain
per entry), find the matching row in the rnapdbee file and compare sequences.
rnaglib is treated as the base/reference.

Output:
  - sequence_mismatches_rnaglib_vs_rnapdbee.csv  — rows where sequences differ
  - summary printed to stdout
"""

import csv
import argparse
from pathlib import Path

TOOL_MISMATCH=['9f9s', '5axm', '7mky', '1e8o', '5ns3', '7d8o', '6prv', '1xjr', '4oqu', '1l2x', '4mgn', '7k16', '5d5l', '4jf2', '6cu1', '5e81', '7n2v', '7zta', '3k1v', '8yup', '5ju8', '6fz0', '5aox', '8v1i', '6mwn', '1y27', '6xko', '8peg', '8g9z']
SEQB_GAPS=['7mlx', '1mms', '5nwq', '5d8h', '5gah', '1il2', '5lzs', '6r5q', '5lys', '6pmo', '3r4f', '3npq', '8am9', '6zym', '8r6c', '8d9k', '5el4', '5ml7', '1l9a', '7p6z', '3kfu', '8s1p', '6mj0', '5ib8', '7d6z', '6wzr', '7mdl', '8k1e']
PLOTTED_PDBIDS=['4v4j', '6gz4', '6ah3', '7o7z', '2xd0', '1mji', '4lvw', '4pr6', '4prf', '3jbu', '3jbv', '4v6d', '7p3k', '5l3p', '8g7p', '3moj', '1fir', '5ccb', '7nwg', '4enc', '2a43', '4lck', '7yse', '4rmo', '6yl5', '8b2l', '5vt0', '6xh2', '3t4b', '7mrl', '5hr6', '6yal', '8cd1', '3d2v', '1et4', '1zci', '1duh', '3lqx', '7qr3', '7qr4', '2der', '5hr7', '6q9a', '1mfq', '3sux', '3wfs', '7o5b', '8h6l', '8k2z', '9dtt', '3gs5', '4p5j', '4xwf', '387d', '7vft', '5mwi', '6e8u', '4v83', '4xej', '5btp', '7kga', '1kh6', '6zmo', '3egz', '4znp', '7b5k', '7eqj', '3rg5', '7zjw', '6vwl', '7ot5']

ANALYZED_PDBIDS=[]
ANALYZED_PDBIDS+=TOOL_MISMATCH
ANALYZED_PDBIDS+=SEQB_GAPS
ANALYZED_PDBIDS+=PLOTTED_PDBIDS

def load_rnapdbee(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Return {(pdbid, chain): (sequence, base_pairs)} from rna_pdb_dataset_bp.csv (rnapdbee)."""
    records: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["id"].strip().lower(), row["chain"].strip())
            records[key] = (row["sequence"].strip(), row["base_pairs"].strip())
    return records


def load_rnaglib(path: Path) -> list[tuple[str, str, str]]:
    """Return [(pdbid, chain, sequence), ...] from sanitized_rnapdbdataset.csv (rnaglib)."""
    records: list[tuple[str, str, str]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            records.append(
                (row["pdbid"].strip().lower(), row["chain"].strip(), row["sequence"].strip())
            )
    return records


def compare(
    rnaglib_path: Path,
    rnapdbee_path: Path,
    out_path: Path,
) -> None:
    rnapdbee = load_rnapdbee(rnapdbee_path)
    rnaglib = load_rnaglib(rnaglib_path)

    mismatches: list[dict] = []
    missing_in_rnapdbee: list[tuple[str, str]] = []

    for pdbid, chain, seq_rnaglib in rnaglib:
        key = (pdbid, chain)
        if key not in rnapdbee:
            missing_in_rnapdbee.append(key)
            continue
        seq_rnapdbee, base_pairs_rnapdbee = rnapdbee[key]
        if seq_rnaglib != seq_rnapdbee:
            if pdbid in ANALYZED_PDBIDS:
                # restrict the analysis to just the PDB IDs
                # we are trying to plot in collab
                mismatches.append(
                    {
                        "pdbid": pdbid,
                        "chain": chain,
                        "sequence_rnaglib": seq_rnaglib,
                        "sequence_rnapdbee": seq_rnapdbee,
                        "base_pairs_rnapdbee": base_pairs_rnapdbee,
                        "len_rnaglib": len(seq_rnaglib),
                        "len_rnapdbee": len(seq_rnapdbee),
                    }
                )

    fieldnames = ["pdbid", "chain", "sequence_rnaglib", "sequence_rnapdbee", "base_pairs_rnapdbee", "len_rnaglib", "len_rnapdbee"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mismatches)

    # Summary
    print(f"RNAglib entries checked         : {len(rnaglib)}")
    print(f"Keys missing in rnapdbee file   : {len(missing_in_rnapdbee)}")
    print(f"Sequence mismatches             : {len(mismatches)}")
    print(f"Output written to               : {out_path}")

    if missing_in_rnapdbee:
        print("\nKeys in rnaglib but not found in rnapdbee:")
        for pdbid, chain in missing_in_rnapdbee:
            print(f"  {pdbid},{chain}")

    if mismatches:
        print("\nMismatches (pdbid, chain, len_rnaglib, len_rnapdbee):")
        for m in mismatches:
            print(f"  {m['pdbid']},{m['chain']}: {m['len_rnaglib']} vs {m['len_rnapdbee']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rnaglib",
        default="sanitized_rnapdbdataset.csv",
        help="Path to the rnaglib CSV, used as base/reference (default: %(default)s)",
    )
    parser.add_argument(
        "--rnapdbee",
        default="rna_pdb_dataset_bp.csv",
        help="Path to the rnapdbee CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="sequence_mismatches_rnaglib_vs_rnapdbee.csv",
        help="Output CSV path (default: %(default)s)",
    )
    args = parser.parse_args()
    compare(Path(args.rnaglib), Path(args.rnapdbee), Path(args.out))


if __name__ == "__main__":
    main()
