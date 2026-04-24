#!/usr/bin/env python3
"""
Generate an HTML page showing cross-chain base pairs per PDB.

Reads crosschain_basepairs.csv and the original rna_pdb_dataset_bp.csv.
For each PDB with cross-chain pairs, displays each chain's sequence and
dot-bracket line-by-line, with cross-chain positions highlighted.
"""

import csv
import html
from collections import OrderedDict


def read_pdb_csv(filepath):
    """Read rna_pdb_dataset_bp.csv, group by PDB id, preserve order."""
    pdb_chains = OrderedDict()
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            pdb_chains.setdefault(row["id"], []).append({
                "chain": row["chain"],
                "sequence": row["sequence"],
                "base_pairs": row["base_pairs"],
            })
    return pdb_chains


def read_crosschain_csv(filepath):
    """Read crosschain_basepairs.csv, group pairs by PDB id."""
    pdb_pairs = OrderedDict()
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            pdb_id = row["pdb_id"]
            pdb_pairs.setdefault(pdb_id, []).append({
                "pos_i": int(row["pos_i"]),
                "chain_i": row["chain_i"],
                "nt_i": row["nt_i"],
                "pos_j": int(row["pos_j"]),
                "chain_j": row["chain_j"],
                "nt_j": row["nt_j"],
                "bracket_i": row["bracket_i"],
                "bracket_j": row["bracket_j"],
                "chain_ranges": row["chain_ranges"],
            })
    return pdb_pairs


def parse_chain_ranges(chain_ranges_str):
    """Parse 'A:1-23;B:24-46' into [(name, start, end), ...]."""
    ranges = []
    for part in chain_ranges_str.split(";"):
        name, span = part.split(":")
        s, e = span.split("-")
        ranges.append((name, int(s), int(e)))
    return ranges


def global_to_local(global_pos, chain_ranges):
    """Convert 1-indexed global position to (chain_name, local_1indexed)."""
    for name, start, end in chain_ranges:
        if start <= global_pos <= end:
            return name, global_pos - start + 1
    return "?", global_pos


def highlight_line(text, positions_set):
    """
    Build an HTML string where characters at 0-indexed positions in
    positions_set are wrapped with a highlight span.
    """
    parts = []
    in_highlight = False
    for i, ch in enumerate(text):
        should_hl = i in positions_set
        if should_hl and not in_highlight:
            parts.append('<span class="hl">')
            in_highlight = True
        elif not should_hl and in_highlight:
            parts.append("</span>")
            in_highlight = False
        parts.append(html.escape(ch))
    if in_highlight:
        parts.append("</span>")
    return "".join(parts)


def generate_html(pdb_chains, pdb_pairs, output_path):
    pdb_ids = list(pdb_pairs.keys())

    lines = [HTML_HEADER.format(count=len(pdb_ids))]

    for idx, pdb_id in enumerate(pdb_ids, 1):
        pairs = pdb_pairs[pdb_id]
        chains = pdb_chains.get(pdb_id, [])
        chain_ranges = parse_chain_ranges(pairs[0]["chain_ranges"])

        # Collect cross-chain global positions per chain
        # chain_name -> set of local 0-indexed positions
        chain_hl = {}
        for cr in chain_ranges:
            chain_hl[cr[0]] = set()

        # Also build a list of pair descriptions for the summary
        pair_summaries = []
        for p in pairs:
            ci, li = global_to_local(p["pos_i"], chain_ranges)
            cj, lj = global_to_local(p["pos_j"], chain_ranges)
            # local positions are 1-indexed; convert to 0-indexed for highlight
            chain_hl.setdefault(ci, set()).add(li - 1)
            chain_hl.setdefault(cj, set()).add(lj - 1)
            pair_summaries.append(
                f'{p["nt_i"]}({ci}:{li}) {p["bracket_i"]}–{p["bracket_j"]} '
                f'{p["nt_j"]}({cj}:{lj})'
            )

        # Deduplicate pair summaries for display
        unique_summaries = list(dict.fromkeys(pair_summaries))
        n_pairs = len(unique_summaries)

        # Header
        chain_names = [cr[0] for cr in chain_ranges]
        lines.append(
            f'<h3>{idx}. <code>{html.escape(pdb_id)}</code> &mdash; '
            f'{n_pairs} cross-chain pair{"s" if n_pairs != 1 else ""}, '
            f'chains: {", ".join(f"<code>{html.escape(c)}</code>" for c in chain_names)}'
            f'</h3>'
        )

        # Chain blocks
        lines.append("<pre>")
        for ch_info in chains:
            cname = ch_info["chain"]
            seq = ch_info["sequence"]
            db = ch_info["base_pairs"]
            hl_positions = chain_hl.get(cname, set())

            label_seq = f"chain {cname} seq"
            label_db = f"chain {cname} db "
            # Pad labels to the same width
            max_label = max(len(label_seq), len(label_db))
            label_seq = label_seq.ljust(max_label)
            label_db = label_db.ljust(max_label)

            seq_html = highlight_line(seq, hl_positions)
            db_html = highlight_line(db, hl_positions)

            lines.append(f"{html.escape(label_seq)} : {seq_html}")
            lines.append(f"{html.escape(label_db)} : {db_html}")
            lines.append("")  # blank line between chains

        lines.append("</pre>")

        # Cross-chain pair list
        lines.append('<details><summary>Cross-chain pairs detail</summary><ul>')
        for s in unique_summaries:
            lines.append(f"<li><code>{html.escape(s)}</code></li>")
        lines.append("</ul></details>")

    lines.append(HTML_FOOTER)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Written {len(pdb_ids)} PDB entries to {output_path}")


HTML_HEADER = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cross-chain base pairs</title>
<style>
  body {{ font-family: sans-serif; font-size: 14px; padding: 2em; }}
  h1   {{ font-family: sans-serif; }}
  h3   {{ font-family: sans-serif; margin-top: 2em; margin-bottom: 0.4em; }}
  pre  {{ font-family: monospace; font-size: 13px; background: #f8f8f8;
          border: 1px solid #ddd; border-radius: 4px; padding: 0.8em 1em;
          line-height: 1.6; overflow-x: auto; margin: 0; }}
  .hl  {{ background: #ffe066; font-weight: bold; }}
  details {{ margin-top: 0.4em; font-size: 0.9em; }}
  summary {{ cursor: pointer; color: #555; }}
  details ul {{ margin-top: 0.3em; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Cross-chain base pairs</h1>
<p>{count} PDB structures with cross-chain base pairs.
   <span style="background:#ffe066;font-weight:bold;padding:2px 6px;border-radius:3px">highlighted</span>
   = nucleotide involved in a cross-chain connection.
</p>"""

HTML_FOOTER = """\
</body>
</html>"""


if __name__ == "__main__":
    pdb_chains = read_pdb_csv("../rna_pdb_dataset_bp.csv")
    pdb_pairs = read_crosschain_csv("crosschain_basepairs.csv")
    generate_html(pdb_chains, pdb_pairs, "crosschain_basepairs.html")
