"""Score every prediction against the full reference, pseudoknots included.

No reference is simplified: MEA cannot emit pseudoknots and eats those pairs as
false negatives, which is exactly what the ThreshKnot panel is there to show.
"""
from ast import literal_eval

import pandas as pd

from common import PREDICTIONS_CSV, SCORES_CSV, dot2bp, f1_strict


def main():
    df = pd.read_csv(PREDICTIONS_CSV, converters={"pred_bp": literal_eval})
    df = df.fillna({"experiment": "", "pred_db": ""})

    refs = {db: dot2bp(db) for db in df.ref_db.unique()}
    scores = [f1_strict(refs[row.ref_db], row.pred_bp) for row in df.itertuples()]

    df["tpr"], df["ppv"], df["f1"] = zip(*scores)
    df["len"] = df.sequence.str.len()
    df["n_ref_bp"] = df.ref_db.map(lambda db: len(refs[db]))
    df["has_pk"] = df.ref_db.str.contains(r"[\[\]{}<>]", regex=True)

    out = df[["unit_id", "kind", "pdb_id", "chain", "experiment", "decoder",
              "n_profiles", "tpr", "ppv", "f1", "len", "n_ref_bp", "has_pk"]]
    out.to_csv(SCORES_CSV, index=False)

    print(out.groupby(["decoder", "kind", "experiment"]).f1
          .agg(["count", "median", "mean"]).round(3).to_string())
    print(f"\nwrote {SCORES_CSV}")


if __name__ == "__main__":
    main()
