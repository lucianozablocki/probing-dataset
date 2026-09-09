"""Shared helpers for the LinearPartition probing evaluation pipeline."""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_CSV = os.path.join(REPO, "structure_and_probing.csv")
LP_BIN = os.path.join(REPO, "LinearPartition", "linearpartition")

OUT = os.path.dirname(os.path.abspath(__file__))
ROWS_CSV = os.path.join(OUT, "rows.csv")
SHAPE_DIR = os.path.join(OUT, "shape")
MANIFEST_CSV = os.path.join(OUT, "manifest.csv")
PREDICTIONS_CSV = os.path.join(OUT, "predictions.csv")
SCORES_CSV = os.path.join(OUT, "scores.csv")

MISSING = -1000  # sentinel for "no reactivity measured" in the source csv

MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "b"],
]
BRACKET_CHARS = set(c for pair in MATCHING_BRACKETS for c in pair)


def is_balanced(struct):
    """True if every bracket page opens and closes cleanly.

    An inter-chain base pair leaves an orphan bracket, so this is what
    identifies the cross-chain rows we drop during preprocessing.
    """
    for xop, xcl in MATCHING_BRACKETS:
        depth = 0
        for x in struct:
            if x == xop:
                depth += 1
            elif x == xcl:
                depth -= 1
                if depth < 0:
                    return False
        if depth != 0:
            return False
    return True


def fold2bp(struct, xop="(", xcl=")"):
    """Base pairs of one bracket page, 1-indexed."""
    openxs = []
    bps = []
    for i, x in enumerate(struct):
        if x == xop:
            openxs.append(i)
        elif x == xcl:
            if not openxs:
                raise ValueError(f"unbalanced {xcl!r} at position {i + 1}: {struct}")
            bps.append([openxs.pop() + 1, i + 1])
    if openxs:
        raise ValueError(f"unclosed {xop!r} at position {openxs[0] + 1}: {struct}")
    return bps


def dot2bp(struct):
    """Base pairs of a dot-bracket string, 1-indexed, all pages.

    Raises on anything malformed. Preprocessing is responsible for keeping
    cross-chain and otherwise unparseable structures out of the pipeline, so a
    failure here means something upstream is wrong, not that a score is missing.
    """
    unknown = set(struct) - BRACKET_CHARS - {"."}
    if unknown:
        raise ValueError(f"unknown characters {sorted(unknown)} in: {struct}")
    bp = []
    for xop, xcl in MATCHING_BRACKETS:
        if xop in struct or xcl in struct:
            bp += fold2bp(struct, xop, xcl)
    return sorted(bp)


def f1_strict(ref_bp, pre_bp):
    """Sensitivity, precision and F1 over exact base-pair matches."""
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = sum(1 for rbp in ref_bp if rbp in pre_bp)
    tp2 = sum(1 for pbp in pre_bp if pbp in ref_bp)
    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)
    return tpr, pre, f1


def project_reactivity(aligned_pdb_seq, reactivity):
    """Move reactivity from the alignment frame to the ungapped PDB frame."""
    if len(aligned_pdb_seq) != len(reactivity):
        raise ValueError(
            f"reactivity has {len(reactivity)} values for an alignment of "
            f"{len(aligned_pdb_seq)} columns"
        )
    return [r for nt, r in zip(aligned_pdb_seq, reactivity) if nt != "-"]


def validate_rows(df):
    """Postconditions on the preprocessed rows.

    Re-checks the drop predicates (catches a misapplied filter or a merge that
    reintroduced rows) and, more usefully, checks invariants no filter enforced.
    """
    assert len(df) > 0, "no rows survived preprocessing"

    # re-checks of the drop predicates
    assert not df.experiment.isna().any(), "row with empty experiment survived"
    assert df.dot_bracket.map(is_balanced).all(), "cross-chain row survived"

    # invariants nothing upstream enforced
    bad_len = df[df.reactivity.map(len) != df.sequence.str.len()]
    assert bad_len.empty, f"{len(bad_len)} rows: projected reactivity != len(sequence)"
    assert (df.dot_bracket.str.len() == df.sequence.str.len()).all(), \
        "dot_bracket length != sequence length"
    assert not df.isna().any().any(), "NaN in preprocessed rows"
    assert df.row_id.is_unique, "row_id is not unique"

    for _, row in df.iterrows():
        bps = dot2bp(row.dot_bracket)  # raises if malformed
        assert bps, f"{row.row_id}: reference has no base pairs"

    # one structure per pdb_id x chain, or the mean and noshape units are ill-defined
    per_struct = df.groupby(["pdb_id", "chain"])
    assert (per_struct.sequence.nunique() == 1).all(), \
        "a pdb_id x chain has more than one sequence"
    assert (per_struct.dot_bracket.nunique() == 1).all(), \
        "a pdb_id x chain has more than one reference structure"


def load_rows():
    from ast import literal_eval

    df = pd.read_csv(ROWS_CSV, converters={"reactivity": literal_eval})
    validate_rows(df)
    return df
