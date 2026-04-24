#!/usr/bin/env python3
"""
Identify cross-chain base pairs in multi-chain PDB structures.

Reads rna_pdb_dataset_bp.csv, concatenates chains per PDB, converts the
combined dot-bracket to base pairs, then checks which pairs span two
different chains.

Output: CSV with columns
  pdb_id, pos_i, chain_i, pos_j, chain_j, bracket_char_i, bracket_char_j
where pos_i/pos_j are 1-indexed within the *full concatenated* sequence,
and chain_i/chain_j are the chain names.
"""
TOOL_MISMATCH=['9f9s', '5axm', '7mky', '1e8o', '5ns3', '7d8o', '6prv', '1xjr', '4oqu', '1l2x', '4mgn', '7k16', '5d5l', '4jf2', '6cu1', '5e81', '7n2v', '7zta', '3k1v', '8yup', '5ju8', '6fz0', '5aox', '8v1i', '6mwn', '1y27', '6xko', '8peg', '8g9z']
SEQB_GAPS=['7mlx', '1mms', '5nwq', '5d8h', '5gah', '1il2', '5lzs', '6r5q', '5lys', '6pmo', '3r4f', '3npq', '8am9', '6zym', '8r6c', '8d9k', '5el4', '5ml7', '1l9a', '7p6z', '3kfu', '8s1p', '6mj0', '5ib8', '7d6z', '6wzr', '7mdl', '8k1e']
PLOTTED_PDBIDS=['4v4j', '6gz4', '6ah3', '7o7z', '2xd0', '1mji', '4lvw', '4pr6', '4prf', '3jbu', '3jbv', '4v6d', '7p3k', '5l3p', '8g7p', '3moj', '1fir', '5ccb', '7nwg', '4enc', '2a43', '4lck', '7yse', '4rmo', '6yl5', '8b2l', '5vt0', '6xh2', '3t4b', '7mrl', '5hr6', '6yal', '8cd1', '3d2v', '1et4', '1zci', '1duh', '3lqx', '7qr3', '7qr4', '2der', '5hr7', '6q9a', '1mfq', '3sux', '3wfs', '7o5b', '8h6l', '8k2z', '9dtt', '3gs5', '4p5j', '4xwf', '387d', '7vft', '5mwi', '6e8u', '4v83', '4xej', '5btp', '7kga', '1kh6', '6zmo', '3egz', '4znp', '7b5k', '7eqj', '3rg5', '7zjw', '6vwl', '7ot5']

ANALYZED_PDBIDS=[]
ANALYZED_PDBIDS+=TOOL_MISMATCH
ANALYZED_PDBIDS+=SEQB_GAPS
ANALYZED_PDBIDS+=PLOTTED_PDBIDS

import csv
import sys
from collections import OrderedDict

# ── Bracket definitions (same as dot2bp.py, with the B/b fix) ──
MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]


def fold2bp(struc, xop="(", xcl=")"):
    """Get base pairs from one page folding (using only one type of brackets).
    BP are 1-indexed"""
    openxs = []
    bps = []
    if struc.count(xop) != struc.count(xcl):
        return False
    for i, x in enumerate(struc):
        if x == xop:
            openxs.append(i)
        elif x == xcl:
            if len(openxs) > 0:
                bps.append([openxs.pop() + 1, i + 1])
            else:
                return False
    return bps


def dot2bp(struct):
    bp = []
    if not set(struct).issubset(
        set(["."] + [c for par in MATCHING_BRACKETS for c in par])
    ):
        return False

    for brackets in MATCHING_BRACKETS:
        if brackets[0] in struct:
            # print(brackets[0], brackets[1])
            bpk = fold2bp(struct, brackets[0], brackets[1])
            if bpk:
                bp = bp + bpk
                # print(list(sorted(bp)))
            else:
                return False
    return list(sorted(bp))


def read_csv(filepath):
    """Read CSV and group chains by PDB id, preserving order."""
    pdb_chains = OrderedDict()
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["id"]
            pdb_chains.setdefault(pdb_id, []).append({
                "chain": row["chain"],
                "sequence": row["sequence"],
                "base_pairs": row["base_pairs"],
            })
    return pdb_chains


def find_crosschain_pairs(pdb_chains):
    """
    For each multi-chain PDB, find base pairs that cross chain boundaries.

    Returns a list of dicts with:
      pdb_id, pos_i, chain_i, nt_i, pos_j, chain_j, nt_j,
      bracket_i, bracket_j, full_sequence, full_dotbracket, chain_ranges
    """
    results = []
    errors = []

    for pdb_id, chains in pdb_chains.items():
        if pdb_id not in ANALYZED_PDBIDS:
            continue
        if len(chains) < 2:
            continue

        # Concatenate sequences and dot-bracket across chains
        full_seq = ""
        full_db = ""
        # chain_ranges: list of (chain_name, start_1indexed, end_1indexed)
        chain_ranges = []
        for ch in chains:
            start = len(full_seq) + 1  # 1-indexed
            full_seq += ch["sequence"]
            full_db += ch["base_pairs"]
            end = len(full_seq)  # 1-indexed
            chain_ranges.append((ch["chain"], start, end))

        # Sanity check lengths
        if len(full_seq) != len(full_db):
            # errors.append(f"{pdb_id}: sequence length ({len(full_seq)}) != "
            #               f"dot-bracket length ({len(full_db)}), skipping")
            print(f"{pdb_id}: sequence length ({len(full_seq)}) != "
                          f"dot-bracket length ({len(full_db)}), skipping")
            raise Exception

        # print(pdb_id)
        # Convert to base pairs
        if pdb_id=='5lzs':
            print("skipping 5LZS, malformed dot bracket")
            # print(full_db)
            continue
        elif pdb_id=='6r5q':
            print("skipping 6R5Q, malformed dot bracket")
            # print(full_db)
            continue
        elif pdb_id=='7o5b':
            print("skipping 7O5B, malformed dot bracket")
            # print(full_db)
            continue
        elif pdb_id=='7o7z':
            print("skipping 7O7Z, malformed dot bracket")
            # print(full_db)
            continue
        # if pdb_id=='7n2v':
        #     print(full_db)
        bps = dot2bp(full_db)
        # print(bps)
        # if not bps:
        #     print(f"{pdb_id}: malformed dot-bracket LUCSI")
        if not bps:
            # errors.append(f"{pdb_id}: malformed dot-bracket, skipping")
            print(f"{pdb_id}: malformed dot-bracket")
            raise Exception

        # Helper: map 1-indexed position to chain name
        def pos_to_chain(pos):
            for cname, cstart, cend in chain_ranges:
                if cstart <= pos <= cend:
                    return cname
            raise Exception("Position out of range")

        # Check each base pair
        cross_pairs = []
        for i, j in bps:
            ci = pos_to_chain(i)
            cj = pos_to_chain(j)
            if ci != cj:
                cross_pairs.append({
                    "pdb_id": pdb_id,
                    "pos_i": i,
                    "chain_i": ci,
                    "nt_i": full_seq[i - 1],
                    "pos_j": j,
                    "chain_j": cj,
                    "nt_j": full_seq[j - 1],
                    "bracket_i": full_db[i - 1],
                    "bracket_j": full_db[j - 1],
                })

        if cross_pairs:
            # Attach context to the first entry for downstream use
            for cp in cross_pairs:
                cp["full_sequence"] = full_seq
                cp["full_dotbracket"] = full_db
                cp["chain_ranges"] = ";".join(
                    f"{c}:{s}-{e}" for c, s, e in chain_ranges
                )
            results.extend(cross_pairs)

    return results, errors


def main():
    csv_path = "../rna_pdb_dataset_bp.csv"
    out_path = "crosschain_basepairs.csv"

    pdb_chains = read_csv(csv_path)
    results, errors = find_crosschain_pairs(pdb_chains)

    # Print errors/warnings
    for e in errors:
        print(f"WARNING: {e}", file=sys.stderr)

    # Summary
    pdb_ids_with_cross = sorted(set(r["pdb_id"] for r in results))
    print(f"Total multi-chain PDBs checked: "
          f"{sum(1 for v in pdb_chains.values() if len(v) > 1)}")
    print(f"PDBs with cross-chain base pairs: {len(pdb_ids_with_cross)}")
    print(f"Total cross-chain base pairs found: {len(results)}")

    # Write CSV
    fieldnames = [
        "pdb_id", "pos_i", "chain_i", "nt_i",
        "pos_j", "chain_j", "nt_j",
        "bracket_i", "bracket_j",
        "chain_ranges", "full_sequence", "full_dotbracket",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Output written to {out_path}")

    # Print first few examples
    print("\n── Sample cross-chain pairs ──")
    seen = set()
    for r in results:
        if r["pdb_id"] not in seen:
            seen.add(r["pdb_id"])
            print(f"  {r['pdb_id']}: pos {r['pos_i']}({r['chain_i']}) "
                  f"{r['nt_i']}{r['bracket_i']} ↔ "
                  f"pos {r['pos_j']}({r['chain_j']}) "
                  f"{r['nt_j']}{r['bracket_j']}")
            if len(seen) >= 15:
                print(f"  ... and {len(pdb_ids_with_cross) - 15} more PDBs")
                break


if __name__ == "__main__":
    main()
