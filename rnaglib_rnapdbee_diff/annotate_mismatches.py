"""
Annotate sequence mismatches from the comparison CSV.

Reads sequence_mismatches_rnaglib_vs_rnapdbee.csv (or --input) and writes:
  - sequence_mismatches_annotated.md   (--md)   — Markdown with bolded diff chars
  - sequence_mismatches_annotated.html (--html) — jsdiff-style colored HTML

rnaglib is the base/reference; rnapdbee is shown second.
The diff uses SequenceMatcher to find the single changed region (substitution
or single-nucleotide insertion/deletion) and marks it prominently.

Usage:
  python annotate_mismatches.py
  python annotate_mismatches.py --md out.md --html out.html
  python annotate_mismatches.py --no-html
"""

import argparse
import csv
import difflib
import html as html_lib
from pathlib import Path


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def get_opcodes(seq1: str, seq2: str) -> list:
    return difflib.SequenceMatcher(None, seq1, seq2, autojunk=False).get_opcodes()


def format_markdown(seq_rnaglib: str, seq_rnapdbee: str) -> tuple[str, str]:
    """Return (formatted_rnaglib, formatted_rnapdbee) with **bold** on differing region."""
    rnaglib_parts, rnapdbee_parts = [], []
    for tag, i1, i2, j1, j2 in get_opcodes(seq_rnaglib, seq_rnapdbee):
        chunk_rnaglib  = seq_rnaglib[i1:i2]
        chunk_rnapdbee = seq_rnapdbee[j1:j2]
        if tag == "equal":
            rnaglib_parts.append(chunk_rnaglib)
            rnapdbee_parts.append(chunk_rnapdbee)
        elif tag == "replace":
            rnaglib_parts.append(f"**{chunk_rnaglib}**")
            rnapdbee_parts.append(f"**{chunk_rnapdbee}**")
        elif tag == "delete":
            rnaglib_parts.append(f"**{chunk_rnaglib}**")
        elif tag == "insert":
            rnapdbee_parts.append(f"**{chunk_rnapdbee}**")
    return "".join(rnaglib_parts), "".join(rnapdbee_parts)


def format_html_spans(seq_rnaglib: str, seq_rnapdbee: str) -> tuple[str, str]:
    """Return (html_rnaglib, html_rnapdbee) with <span class="diff"> on differing region."""
    rnaglib_parts, rnapdbee_parts = [], []
    for tag, i1, i2, j1, j2 in get_opcodes(seq_rnaglib, seq_rnapdbee):
        chunk_rnaglib  = html_lib.escape(seq_rnaglib[i1:i2])
        chunk_rnapdbee = html_lib.escape(seq_rnapdbee[j1:j2])
        if tag == "equal":
            rnaglib_parts.append(chunk_rnaglib)
            rnapdbee_parts.append(chunk_rnapdbee)
        elif tag == "replace":
            rnaglib_parts.append(f'<span class="sub">{chunk_rnaglib}</span>')
            rnapdbee_parts.append(f'<span class="sub">{chunk_rnapdbee}</span>')
        elif tag == "delete":
            rnaglib_parts.append(f'<span class="del">{chunk_rnaglib}</span>')
        elif tag == "insert":
            rnapdbee_parts.append(f'<span class="ins">{chunk_rnapdbee}</span>')
    return "".join(rnaglib_parts), "".join(rnapdbee_parts)


def diff_type(seq_rnaglib: str, seq_rnapdbee: str) -> str:
    tags = {op[0] for op in get_opcodes(seq_rnaglib, seq_rnapdbee) if op[0] != "equal"}
    if tags == {"replace"}:
        return "substitution"
    if "delete" in tags:
        return "deletion in rnapdbee"
    if "insert" in tags:
        return "insertion in rnapdbee"
    return "unknown"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Sequence mismatches: rnaglib (base) vs rnapdbee",
        "",
        f"{len(rows)} mismatches. Bold marks the differing nucleotide(s).",
        "",
    ]
    for i, row in enumerate(rows, 1):
        pdbid        = row["pdbid"]
        chain        = row["chain"]
        seq_rnaglib  = row["sequence_rnaglib"]
        seq_rnapdbee = row["sequence_rnapdbee"]
        base_pairs   = row.get("base_pairs_rnapdbee", "")
        dtype        = diff_type(seq_rnaglib, seq_rnapdbee)
        fmt_rnaglib, fmt_rnapdbee = format_markdown(seq_rnaglib, seq_rnapdbee)

        lines += [
            "---",
            "",
            f"### {i}. `{pdbid}` chain `{chain}`  —  {dtype}",
            "",
            f"| | Sequence | len |",
            f"|---|---|---|",
            f"| **rnaglib** | {fmt_rnaglib} | {len(seq_rnaglib)} |",
            f"| **rnapdbee** | {fmt_rnapdbee} | {len(seq_rnapdbee)} |",
            f"| **dot-bracket** | `{base_pairs}` | |",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown written to: {out_path}")


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sequence mismatches</title>
<style>
  body {{ font-family: monospace; font-size: 14px; padding: 2em; }}
  h1   {{ font-family: sans-serif; }}
  h3   {{ font-family: sans-serif; margin-top: 2em; }}
  table {{ border-collapse: collapse; margin-bottom: 1em; }}
  td, th {{ padding: 4px 10px; border: 1px solid #ccc; }}
  th {{ background: #f0f0f0; }}
  .sub {{ background: #ffe066; font-weight: bold; border-radius: 2px; }}
  .del {{ background: #ff9999; font-weight: bold; border-radius: 2px; }}
  .ins {{ background: #99ff99; font-weight: bold; border-radius: 2px; }}
  .legend span {{ display: inline-block; padding: 2px 8px; border-radius: 3px; margin-right: 8px; }}
</style>
</head>
<body>
<h1>Sequence mismatches: rnaglib (base) vs rnapdbee</h1>
<p>{count} mismatches.</p>
<p class="legend">
  Legend:&nbsp;
  <span class="sub">substitution</span>
  <span class="del">deletion&nbsp;(extra in rnaglib)</span>
  <span class="ins">insertion&nbsp;(extra in rnapdbee)</span>
</p>
{body}
</body>
</html>
"""

def write_html(rows: list[dict], out_path: Path) -> None:
    body_parts = []
    for i, row in enumerate(rows, 1):
        pdbid        = row["pdbid"]
        chain        = row["chain"]
        seq_rnaglib  = row["sequence_rnaglib"]
        seq_rnapdbee = row["sequence_rnapdbee"]
        base_pairs   = html_lib.escape(row.get("base_pairs_rnapdbee", ""))
        dtype        = diff_type(seq_rnaglib, seq_rnapdbee)
        fmt_rnaglib, fmt_rnapdbee = format_html_spans(seq_rnaglib, seq_rnapdbee)

        body_parts.append(
            f"<h3>{i}. <code>{html_lib.escape(pdbid)}</code> "
            f"chain <code>{html_lib.escape(chain)}</code> &mdash; {dtype}</h3>\n"
            f"<table>\n"
            f"  <tr><th></th><th>Sequence</th><th>len</th></tr>\n"
            f"  <tr><td><b>rnaglib</b></td><td>{fmt_rnaglib}</td><td>{len(seq_rnaglib)}</td></tr>\n"
            f"  <tr><td><b>rnapdbee</b></td><td>{fmt_rnapdbee}</td><td>{len(seq_rnapdbee)}</td></tr>\n"
            f"  <tr><td><b>dot-bracket</b></td><td><code>{base_pairs}</code></td><td></td></tr>\n"
            f"</table>"
        )

    out_path.write_text(
        HTML_TEMPLATE.format(count=len(rows), body="\n".join(body_parts)),
        encoding="utf-8",
    )
    print(f"HTML written to:     {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="sequence_mismatches_rnaglib_vs_rnapdbee.csv",
        help="Mismatch CSV from compare_sequences.py (default: %(default)s)",
    )
    parser.add_argument(
        "--md",
        default="sequence_mismatches_annotated.md",
        help="Output Markdown path (default: %(default)s)",
    )
    parser.add_argument(
        "--html",
        default="sequence_mismatches_annotated.html",
        help="Output HTML path (default: %(default)s)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML output",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        help="Skip Markdown output",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    with Path(args.input).open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"Loaded {len(rows)} mismatches from {args.input}")

    if not args.no_md:
        write_markdown(rows, Path(args.md))
    if not args.no_html:
        write_html(rows, Path(args.html))


if __name__ == "__main__":
    main()
