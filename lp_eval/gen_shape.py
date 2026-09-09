"""Write one .shape file per evaluation unit, plus the manifest run_lp reads.

Three kinds of unit:
  row      one probing profile, as measured
  mean     one structure x experiment, profiles averaged position-wise
  noshape  one structure, folded with no restraint at all

Values go in verbatim. LinearPartition treats every negative value as no-data
(LinearPartition.cpp:574), so the -1000 sentinel needs no conversion. It does
NOT honour the "leave the line out" half of the RNAstructure SHAPE convention:
it reads the position column and then ignores it, indexing values by file
order, so every position 1..L must be present or the whole profile shifts.
"""
import os

import numpy as np
import pandas as pd

from common import MANIFEST_CSV, MISSING, SHAPE_DIR, load_rows

# DMS methylates N1-A and N3-C only, so its G and U values are background
# rather than structural signal. They are masked to the missing sentinel (a
# negative value, which LinearPartition reads as no-data) and NOT to zero,
# which it would read as a pairing bonus. 2A3 reacts at all four
# nucleotides and is left alone.
DMS_REACTIVE = {"A", "C"}


def mask_dms(sequence, reactivity, experiment):
    """Blank out the nucleotides DMS cannot report on."""
    if not experiment.startswith("DMS"):
        return list(reactivity)
    return [v if nt in DMS_REACTIVE else MISSING
            for nt, v in zip(sequence, reactivity)]


def write_shape(path, values):
    with open(path, "w") as fh:
        for pos, val in enumerate(values, start=1):
            fh.write(f"{pos}\t{val}\n")
    check_shape(path, len(values))


def check_shape(path, length):
    """Every position 1..L present, in order. This is the invariant whose
    violation silently shifts a profile rather than raising."""
    with open(path) as fh:
        positions = [int(line.split()[0]) for line in fh if line.strip()]
    assert positions == list(range(1, length + 1)), \
        f"{path}: positions are not 1..{length} in order"


def mean_profile(profiles):
    """Position-wise mean, skipping the missing sentinel."""
    arr = np.array([list(p) for p in profiles], dtype=float)
    arr[arr == MISSING] = np.nan
    all_missing = np.isnan(arr).all(axis=0)
    mean = np.full(arr.shape[1], float(MISSING))
    mean[~all_missing] = np.nanmean(arr[:, ~all_missing], axis=0)
    return [MISSING if m else round(float(v), 6) for v, m in zip(mean, all_missing)]


def main():
    df = load_rows()
    df["reactivity"] = [
        mask_dms(r.sequence, r.reactivity, r.experiment) for r in df.itertuples()
    ]
    os.makedirs(SHAPE_DIR, exist_ok=True)
    units = []

    for row in df.itertuples():
        path = os.path.join(SHAPE_DIR, f"{row.row_id}.shape")
        write_shape(path, row.reactivity)
        units.append(dict(
            unit_id=row.row_id, kind="row", pdb_id=row.pdb_id, chain=row.chain,
            experiment=row.experiment, shape_path=path, n_profiles=1,
            sequence=row.sequence, ref_db=row.dot_bracket,
        ))

    for (pdb_id, chain, experiment), grp in df.groupby(["pdb_id", "chain", "experiment"]):
        unit_id = f"{pdb_id}_{chain}_{experiment}_mean"
        path = os.path.join(SHAPE_DIR, f"{unit_id}.shape")
        write_shape(path, mean_profile(grp.reactivity.tolist()))
        units.append(dict(
            unit_id=unit_id, kind="mean", pdb_id=pdb_id, chain=chain,
            experiment=experiment, shape_path=path, n_profiles=len(grp),
            sequence=grp.sequence.iloc[0], ref_db=grp.dot_bracket.iloc[0],
        ))

    for (pdb_id, chain), grp in df.groupby(["pdb_id", "chain"]):
        # no reactivity to store: this entry exists to tell run_lp to fold the
        # sequence with no --shape argument
        units.append(dict(
            unit_id=f"{pdb_id}_{chain}_noshape", kind="noshape", pdb_id=pdb_id,
            chain=chain, experiment="", shape_path="", n_profiles=0,
            sequence=grp.sequence.iloc[0], ref_db=grp.dot_bracket.iloc[0],
        ))

    manifest = pd.DataFrame(units)
    assert manifest.unit_id.is_unique, "unit_id is not unique"
    manifest.to_csv(MANIFEST_CSV, index=False)

    print(manifest.kind.value_counts().to_string())
    print(f"\n{len(manifest)} units -> {2 * len(manifest)} LinearPartition runs")
    print(f"wrote {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
