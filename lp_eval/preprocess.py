"""Filter structure_and_probing.csv down to the rows we can fairly evaluate.

Drops, in order:
  frame mismatch    reactivity is indexed on the alignment; if the alignment is
                    longer than the pdb sequence the two frames disagree
  cross-chain       an inter-chain base pair leaves an orphan bracket
  empty experiment  unusable for a DMS vs 2A3 comparison
  zero-pair ref     f1_strict returns 1.0 for two empty pair lists, so an
                    unpaired reference scores noise rather than signal
"""
from ast import literal_eval

import pandas as pd

from common import ROWS_CSV, SOURCE_CSV, project_reactivity, is_balanced, validate_rows


def main():
    df = pd.read_csv(SOURCE_CSV, converters={"reactivity": literal_eval})
    df = df.reset_index().rename(columns={"index": "csv_index"})
    n_in = len(df)
    drops = {}

    def drop(name, keep_mask):
        nonlocal df
        drops[name] = int((~keep_mask).sum())
        df = df[keep_mask].copy()

    drop("frame mismatch", df.aligned_pdb_seq.str.len() == df.sequence.str.len())
    drop("cross-chain", df.dot_bracket.map(is_balanced))
    drop("empty experiment", df.experiment.notna())
    drop("zero-pair reference", df.dot_bracket.str.count(r"\(") > 0)

    df["reactivity"] = [
        project_reactivity(row.aligned_pdb_seq, row.reactivity)
        for row in df.itertuples()
    ]
    df["row_id"] = (
        df.pdb_id + "_" + df.chain + "_" + df.experiment + "_" + df.csv_index.astype(str)
    )
    out = df[
        ["row_id", "csv_index", "pdb_id", "chain", "experiment",
         "sequence", "dot_bracket", "reactivity"]
    ]

    # the individual asserts all pass even when two filters claim the same row,
    # which would make the waterfall printed below a lie
    assert len(out) == n_in - sum(drops.values()), "drop counts do not add up"
    validate_rows(out)

    out.to_csv(ROWS_CSV, index=False)

    width = max(len(k) for k in drops)
    print(f"{'input':>{width}}  {n_in:>5}")
    for name, n in drops.items():
        print(f"{name:>{width}}  {-n:>5}")
    print(f"{'kept':>{width}}  {len(out):>5}")
    print()
    print(f"structures (pdb_id x chain): {out.groupby(['pdb_id', 'chain']).ngroups}")
    print(f"mean units (structure x experiment): "
          f"{out.groupby(['pdb_id', 'chain', 'experiment']).ngroups}")
    for exp, n in out.experiment.value_counts().items():
        print(f"  {exp}: {n}")
    print(f"\nwrote {ROWS_CSV}")


if __name__ == "__main__":
    main()
