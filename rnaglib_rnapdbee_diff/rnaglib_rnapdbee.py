import pandas as pd
TOOL_MISMATCH=['9f9s', '5axm', '7mky', '1e8o', '5ns3', '7d8o', '6prv', '1xjr', '4oqu', '1l2x', '4mgn', '7k16', '5d5l', '4jf2', '6cu1', '5e81', '7n2v', '7zta', '3k1v', '8yup', '5ju8', '6fz0', '5aox', '8v1i', '6mwn', '1y27', '6xko', '8peg', '8g9z']
print(len(TOOL_MISMATCH))
mismatches=pd.read_csv("./sequence_mismatches_bp_vs_sanitized.csv")

mismatches_ids=mismatches['pdbid'].tolist()
print(len(set(mismatches_ids)))
for m in mismatches_ids:
    if m not in TOOL_MISMATCH:
        print(m)

"""
29 -> unique PDB IDs not plotted by collab
34 -> unique PDB IDs with different sequences between rnaglib and rnapdbee
5el4 -> bug in collab (because of not "ungrouping" the chains)
5l3p -> Y differs, B not (and this was the aligned sequence)
8g7p -> 
8k1e -> B differs, A not (and this was the aligned sequence)
8r6c ->
"""