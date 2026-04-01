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


def tag_seqs(
    seq_rnaglib: str, seq_rnapdbee: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Tag differing regions in each sequence without any padding.

    Returns:
        rng_parts — list of (kind, text) for rnaglib
        rpd_parts — list of (kind, text) for rnapdbee

    kind values: "equal" | "diff"
    """
    rng_parts: list[tuple[str, str]] = []
    rpd_parts: list[tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in get_opcodes(seq_rnaglib, seq_rnapdbee):
        chunk_rng = seq_rnaglib[i1:i2]
        chunk_rpd = seq_rnapdbee[j1:j2]
        if tag == "equal":
            rng_parts.append(("equal", chunk_rng))
            rpd_parts.append(("equal", chunk_rpd))
        else:  # replace, delete, insert
            if chunk_rng:
                rng_parts.append(("diff", chunk_rng))
            if chunk_rpd:
                rpd_parts.append(("diff", chunk_rpd))

    return rng_parts, rpd_parts


_DIFF_STYLE = 'background:#ffe066;font-weight:bold'


def render_tagged(parts: list[tuple[str, str]]) -> str:
    """
    Render (kind, text) parts into an HTML string for use inside a <pre> block.
    Diff positions get a yellow highlight.
    """
    out = []
    for kind, text in parts:
        t = html_lib.escape(text)
        if kind == "diff":
            out.append(f'<span style="{_DIFF_STYLE}">{t}</span>')
        else:
            out.append(t)
    return "".join(out)


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
        f"{len(rows)} mismatches.",
        "Highlighted nucleotide(s) mark the diff.",
        "",
    ]
    for i, row in enumerate(rows, 1):
        pdbid        = row["pdbid"]
        chain        = row["chain"]
        seq_rnaglib  = row["sequence_rnaglib"]
        seq_rnapdbee = row["sequence_rnapdbee"]
        base_pairs   = row.get("base_pairs_rnapdbee", "")
        dtype        = diff_type(seq_rnaglib, seq_rnapdbee)

        rng_parts, rpd_parts = tag_seqs(seq_rnaglib, seq_rnapdbee)
        tagged_rng = render_tagged(rng_parts)
        tagged_rpd = render_tagged(rpd_parts)
        escaped_bp = html_lib.escape(base_pairs)

        lines += [
            "---",
            "",
            f"### {i}. `{pdbid}` chain `{chain}`  —  {dtype}",
            "",
            "<pre>",
            f"rnaglib  : {tagged_rng}",
            f"rnapdbee : {tagged_rpd}",
            f"dot-br   : {escaped_bp}",
            "</pre>",
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
  body {{ font-family: sans-serif; font-size: 14px; padding: 2em; }}
  h1   {{ font-family: sans-serif; }}
  h3   {{ font-family: sans-serif; margin-top: 2em; margin-bottom: 0.4em; }}
  pre  {{ font-family: monospace; font-size: 13px; background: #f8f8f8;
          border: 1px solid #ddd; border-radius: 4px; padding: 0.8em 1em;
          line-height: 1.6; overflow-x: auto; margin: 0; }}
  .legend span {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
                  margin-right: 8px; font-family: monospace; }}
</style>
</head>
<body>
<h1>Sequence mismatches: rnaglib (base) vs rnapdbee</h1>
<p>{count} mismatches.
   <span style="background:#ffe066;font-weight:bold;padding:2px 6px;border-radius:3px">highlighted</span>
   = diff nucleotide(s)
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
        base_pairs   = row.get("base_pairs_rnapdbee", "")
        dtype        = diff_type(seq_rnaglib, seq_rnapdbee)

        rng_parts, rpd_parts = tag_seqs(seq_rnaglib, seq_rnapdbee)
        tagged_rng = render_tagged(rng_parts)
        tagged_rpd = render_tagged(rpd_parts)
        escaped_bp = html_lib.escape(base_pairs)

        body_parts.append(
            f"<h3>{i}. <code>{html_lib.escape(pdbid)}</code> "
            f"chain <code>{html_lib.escape(chain)}</code> &mdash; {dtype}</h3>\n"
            f"<pre>"
            f"rnaglib  : {tagged_rng}\n"
            f"rnapdbee : {tagged_rpd}\n"
            f"dot-br   : {escaped_bp}"
            f"</pre>"
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
        default="rnaglib_rnapdbee_diff/sequence_mismatches_rnaglib_vs_rnapdbee.csv",
        help="Mismatch CSV from compare_sequences.py (default: %(default)s)",
    )
    parser.add_argument(
        "--md",
        default="rnaglib_rnapdbee_diff/sequence_mismatches_annotated.md",
        help="Output Markdown path (default: %(default)s)",
    )
    parser.add_argument(
        "--html",
        default="rnaglib_rnapdbee_diff/sequence_mismatches_annotated.html",
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
