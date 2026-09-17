"""Invariants structure_and_probing.csv must satisfy. Any failure is a bug upstream.

Everything here is a postcondition of postprocess_structure_and_probing.py, which
is the step that collapses the three frames in the raw csv (the full rnagym read,
the gap-padded alignment, the ungapped pdb sequence) into one.
"""
from ast import literal_eval

import pandas as pd
from sklearn.metrics import roc_auc_score

CSV = "structure_and_probing.csv"
KEY = ["pdb_id", "rnagym_id", "experiment", "chain"]
MISSING = -1000

LENGTH_COLUMNS = ["reactivity", "reactivity_errors", "aligned_pdb_seq",
                  "aligned_rnagym_seq", "sequence", "dot_bracket"]


def find_alignment_bounds(alignment_seq):
    idx = [i for i, c in enumerate(alignment_seq) if c != "-"]
    return (idx[0], idx[-1]) if idx else (None, None)


def check_keys(df):
    """<pdb_id, rnagym_id, experiment, chain> identifies a row.

    Duplicates mean two sources emitted the same alignment -- the failure that
    put a half-corrected 5aox into the dataset.
    """
    dupes = df[df.duplicated(subset=KEY, keep=False)]
    assert dupes.empty, \
        f"{len(dupes)} duplicated rows, e.g. {dupes.iloc[0][KEY].to_dict()}"
    return f"{len(df)} rows, all keys unique"


def check_lengths(df):
    """One frame: position i means the same nucleotide in every column.

    aligned_rnagym_seq is included deliberately -- it is edited by the gap
    removal, not just cropped, so it can drift out of step on its own.
    """
    lengths = pd.DataFrame({c: (df[c].map(len) if df[c].dtype == object and
                                isinstance(df[c].iloc[0], list) else df[c].str.len())
                            for c in LENGTH_COLUMNS})
    bad = lengths[lengths.nunique(axis=1) != 1]
    assert bad.empty, \
        f"{len(bad)} rows where the columns disagree on length, e.g.\n{bad.head(3)}"
    return f"all {len(LENGTH_COLUMNS)} columns agree on length in every row"


def check_pdb_frame(df):
    """The alignment's pdb side must be exactly the structure's sequence.

    No interior gaps (remove_gap.py / remove_remaining_gaps.py remove them) and
    no content disagreement (the rnaglib-vs-rnapdbee deletions resolve those).
    """
    # the more specific failure first: a surviving gap points at the gap removal,
    # a content disagreement points at the rnaglib/rnapdbee deletions. The content
    # check below would catch a gap too, but would misname it.
    gapped = df[df.aligned_pdb_seq.str.contains("-")]
    assert gapped.empty, \
        f"{len(gapped)} rows still carry a pdb gap, " \
        f"e.g. {gapped.iloc[0].pdb_id}_{gapped.iloc[0].chain}"
    ungapped = df.aligned_pdb_seq.str.replace("-", "", regex=False)
    bad = df[ungapped != df.sequence]
    assert bad.empty, \
        f"{len(bad)} rows whose alignment disagrees with the structure sequence, " \
        f"e.g. {bad.iloc[0].pdb_id}_{bad.iloc[0].chain}"
    return "pdb side of every alignment equals its structure sequence, gap-free"


def check_rnagym_sentinels(df):
    """A rnagym gap means the residue exists but was not probed.

    The column stays (the structure needs it) and the measurement is MISSING.
    """
    rows = df[df.aligned_rnagym_seq.str.contains("-")]
    bad = 0
    for r in rows.itertuples():
        for nuc, value in zip(r.aligned_rnagym_seq, r.reactivity):
            if nuc == "-" and value != MISSING:
                bad += 1
    assert bad == 0, f"{bad} rnagym-gap columns not marked {MISSING}"
    return f"{len(rows)} rows have rnagym gaps; every gap column marked {MISSING}"


def dms_auc(sub, shift=0):
    """How well reactivity separates the nucleotides DMS reacts with (A/C) from
    the ones it does not (G/U).

    DMS methylates N1-A and N3-C only, so on correctly framed data A/C values
    rank above G/U. Shifting the array by one attributes each value to its
    neighbour, which is near-random with respect to A/C, so the separation
    collapses toward 0.5. That makes this an alignment check that needs no
    ground truth beyond the sequence itself. 2A3 reacts at all four
    nucleotides, so it has nothing to separate on and is excluded.
    """
    values, labels = [], []
    for r in sub.itertuples():
        for i, nuc in enumerate(r.aligned_rnagym_seq):
            j = i + shift
            if nuc == "-" or not 0 <= j < len(r.reactivity) or r.reactivity[j] <= -500:
                continue
            values.append(r.reactivity[j])
            labels.append(int(nuc.upper() in "AC"))
    if len(set(labels)) < 2 or len(values) < MIN_VALUES:
        return None
    return roc_auc_score(labels, values)


MIN_VALUES = 2000   # below this the auc is too noisy to judge a peak by
AUC_FLOOR = 0.65    # the dataset sits near 0.77; a frame error lands near 0.5
SHIFTS = [-2, -1, 0, 1, 2]


def check_dms_frame(df):
    """Reactivity must be framed correctly, globally and per structure."""
    dms = df[df.experiment == "DMS_MaP"]
    aucs = {s: dms_auc(dms, s) for s in SHIFTS}
    assert aucs[0] is not None and aucs[0] >= AUC_FLOOR, \
        f"DMS A/C vs G/U auc is {aucs[0]}, below {AUC_FLOOR}: reactivity looks misframed"
    assert aucs[0] == max(v for v in aucs.values() if v is not None), \
        f"DMS auc peaks off zero, so reactivity is shifted: " \
        f"{ {s: round(v, 3) for s, v in aucs.items() if v} }"

    offenders = []
    for pdb_id, group in dms.groupby("pdb_id"):
        per = {s: dms_auc(group, s) for s in SHIFTS}
        if per[0] is None:
            continue
        if per[0] != max(v for v in per.values() if v is not None):
            offenders.append((pdb_id, {s: round(v, 3) for s, v in per.items() if v}))
    assert not offenders, f"auc peaks off zero for {offenders}"
    return (f"DMS auc {aucs[0]:.3f} at shift 0 "
            f"({aucs[-1]:.3f} / {aucs[1]:.3f} at -1 / +1), peak at 0 for every pdb_id")


def main():
    df = pd.read_csv(CSV, converters={"reactivity": literal_eval,
                                      "reactivity_errors": literal_eval})
    for check in (check_keys, check_lengths, check_pdb_frame,
                  check_rnagym_sentinels, check_dms_frame):
        print(f"ok  {check.__name__:24} {check(df)}")


if __name__ == "__main__":
    main()
