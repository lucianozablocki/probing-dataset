"""Fold every manifest unit with both decoders.

MEA (-M) emits a dot-bracket string; ThreshKnot (-T) emits bpseq. Both are
normalised to a base-pair list so scoring goes through one path. Serial, and
anything unexpected raises rather than being recorded as a failed run.
"""
import subprocess

import pandas as pd

from common import BRACKET_CHARS, LP_BIN, MANIFEST_CSV, PREDICTIONS_CSV, dot2bp

THRESHKNOT_THRESHOLD = "0.3"


def run(sequence, shape_path, decoder):
    cmd = [LP_BIN, "-V"]
    if decoder == "MEA":
        cmd += ["-M"]
    else:
        cmd += ["-T", "--threshold", THRESHKNOT_THRESHOLD]
    if shape_path:
        cmd += ["--shape", shape_path]

    proc = subprocess.run(
        cmd, input=sequence + "\n", capture_output=True, text=True, check=True
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if decoder == "MEA":
        return parse_mea(lines, sequence)
    return parse_bpseq(lines, sequence)


def parse_mea(lines, sequence):
    struct = lines[-1].strip()
    if len(struct) != len(sequence) or not set(struct) <= (BRACKET_CHARS | {"."}):
        raise ValueError(f"unparseable MEA output: {struct!r}")
    return dot2bp(struct), struct


def parse_bpseq(lines, sequence):
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[2].lstrip("-").isdigit():
            rows.append((int(parts[0]), parts[1], int(parts[2])))
    if [i for i, _, _ in rows] != list(range(1, len(sequence) + 1)):
        raise ValueError(f"bpseq output is not 1..{len(sequence)} in order")
    # bpseq lists each pair from both sides; keep it once
    return sorted([i, j] for i, _, j in rows if 0 < i < j), ""


def main():
    manifest = pd.read_csv(MANIFEST_CSV).fillna({"shape_path": "", "experiment": ""})
    out = []
    total = 2 * len(manifest)

    for n, unit in enumerate(manifest.itertuples(), start=1):
        for decoder in ("MEA", "PK"):
            pred_bp, pred_db = run(unit.sequence, unit.shape_path, decoder)
            out.append(dict(
                unit_id=unit.unit_id, kind=unit.kind, pdb_id=unit.pdb_id,
                chain=unit.chain, experiment=unit.experiment, decoder=decoder,
                n_profiles=unit.n_profiles, sequence=unit.sequence,
                ref_db=unit.ref_db, pred_bp=pred_bp, pred_db=pred_db,
            ))
        if n % 250 == 0:
            print(f"{2 * n}/{total} runs", flush=True)

    df = pd.DataFrame(out)
    assert len(df) == total
    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"wrote {PREDICTIONS_CSV} ({len(df)} predictions)")


if __name__ == "__main__":
    main()
