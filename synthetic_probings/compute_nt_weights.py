#!/usr/bin/env python3
"""Compute nucleotide weight vectors from nts_at_which_max_occurs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np

def histogram_values_weights(
    values: list[float],
    bins: int = 40,
) -> tuple[list[float], list[float]]:
    counts, bin_edges = np.histogram(values, bins=bins)
    nonzero_mask = counts > 0

    # Use bin centers as representative values for each histogram bin.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    filtered_values = bin_centers[nonzero_mask]
    filtered_counts = counts[nonzero_mask]

    total_filtered = filtered_counts.sum()
    if total_filtered == 0:
        return [], []

    filtered_weights = filtered_counts / total_filtered
    rounded_values = [round(v, 4) for v in filtered_values.tolist()]
    rounded_weights = [round(w, 4) for w in filtered_weights.tolist()]
    return rounded_values, rounded_weights

def weights_from_nt_list(nt_list: list[str], nt_order: tuple[str, ...] = ("A", "C", "G", "U")) -> list[float]:
    counts = Counter(nt_list)
    print(counts)
    total = len(nt_list)
    if total == 0:
        raise ValueError("Input nt list is empty; cannot compute weights.")
    return [round(counts[nt] / total, 2) for nt in nt_order]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute A/C/G/U weights for each experiment in nts_at_which_max_occurs."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="synthetic_probings/synthetic_probing_results_N5_fixdms.json",
        help="Path to JSON file (default: synthetic_probings/synthetic_probing_results_N5.json)",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nt_occurs = data["nts_at_which_max_occurs"]
    max_by_nt = data["max_by_nt"]
    nt_order = ("A", "C", "G", "U")

    dms_weights = weights_from_nt_list(nt_occurs["DMS_MaP"])
    twoa3_weights = weights_from_nt_list(nt_occurs["2A3_MaP"])

    print(f"DMS_MaP_weights={dms_weights}")
    print(f"2A3_MaP_weights={twoa3_weights}")

    for experiment in ("DMS_MaP", "2A3_MaP"):
        print(f"\n{experiment} value/weight lists by nt:")
        exp_values = max_by_nt[experiment]
        for nt in nt_order:
            nt_values = exp_values[nt]
            values_list, weights_list = histogram_values_weights(nt_values)
            print(f"{experiment}_{nt}_values={values_list}")
            print(f"{experiment}_{nt}_weights={weights_list}")


if __name__ == "__main__":
    main()
