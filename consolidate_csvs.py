import pandas as pd

TOOL_MISMATCH_PATH="tool_mismatch.csv"
PDB_GAPS_PATH="alignments_seqb_gaps_removed.csv"
NO_MODIFIED_PATH="alignments_rows_completed.csv"

TOOL_MISMATCH=['9f9s', '5axm', '7mky', '1e8o', '5ns3', '7d8o', '6prv', '1xjr', '4oqu', '1l2x', '4mgn', '7k16', '5d5l', '4jf2', '6cu1', '5e81', '7n2v', '7zta', '3k1v', '8yup', '5ju8', '6fz0', '5aox', '8v1i', '6mwn', '1y27', '6xko', '8peg', '8g9z']
SEQB_GAPS=['7mlx', '1mms', '5nwq', '5d8h', '5gah', '1il2', '5lzs', '6r5q', '5lys', '6pmo', '3r4f', '3npq', '8am9', '6zym', '8r6c', '8d9k', '5el4', '5ml7', '1l9a', '7p6z', '3kfu', '8s1p', '6mj0', '5ib8', '7d6z', '6wzr', '7mdl', '8k1e']
PLOTTED_PDBIDS=['4v4j', '6gz4', '6ah3', '7o7z', '2xd0', '1mji', '4lvw', '4pr6', '4prf', '3jbu', '3jbv', '4v6d', '7p3k', '5l3p', '8g7p', '3moj', '1fir', '5ccb', '7nwg', '4enc', '2a43', '4lck', '7yse', '4rmo', '6yl5', '8b2l', '5vt0', '6xh2', '3t4b', '7mrl', '5hr6', '6yal', '8cd1', '3d2v', '1et4', '1zci', '1duh', '3lqx', '7qr3', '7qr4', '2der', '5hr7', '6q9a', '1mfq', '3sux', '3wfs', '7o5b', '8h6l', '8k2z', '9dtt', '3gs5', '4p5j', '4xwf', '387d', '7vft', '5mwi', '6e8u', '4v83', '4xej', '5btp', '7kga', '1kh6', '6zmo', '3egz', '4znp', '7b5k', '7eqj', '3rg5', '7zjw', '6vwl', '7ot5']
PDB_IDS=[]
PDB_IDS+=TOOL_MISMATCH
PDB_IDS+=SEQB_GAPS
PDB_IDS+=PLOTTED_PDBIDS

# read all csvs
tool_mismatch_df=pd.read_csv(TOOL_MISMATCH_PATH)
pdb_gaps_df=pd.read_csv(PDB_GAPS_PATH)
no_modified_df=pd.read_csv(NO_MODIFIED_PATH)
# print(f"len tool mismatch: {len(tool_mismatch_df)}")
# print(f"len pdb gaps: {len(pdb_gaps_df)}")
# print(f"len no modified: {len(no_modified_df)}")

set_1=set(tool_mismatch_df.columns)
set_2=set(pdb_gaps_df.columns)
set_3=set(no_modified_df.columns)

assert(set_1==set_2==set_3)

# concatenate dfs
# count how many pdb_id,chain are there
# count how many rows

res=pd.concat([tool_mismatch_df, no_modified_df, pdb_gaps_df], ignore_index=True)
print(f"probing and structure csv has {len(res)} rows")
print(f"and {len(res.groupby(['pdb_id','chain']))} unique structures")

included_pdb_ids = res['pdb_id'].unique()

tool_mismatch_but_no_alignment_modified=[]
for pdb_id in PDB_IDS:
    if pdb_id not in included_pdb_ids:
        # print(f"{pdb_id} not included")
        tool_mismatch_but_no_alignment_modified.append(pdb_id)
# print(tool_mismatch_but_no_alignment_modified)
assert (len(tool_mismatch_but_no_alignment_modified)==0)

# out_df = pd.DataFrame(results)
# out_df['reactivity'] = out_df['reactivity'].apply(lambda x: json.dumps(x.tolist()))
# out_df['reactivity_errors'] = out_df['reactivity_errors'].apply(lambda x: json.dumps(x.tolist()))
res.to_csv("structure_and_probing.csv", index=False)
print(f"Written {len(res)} rows to structure_and_probing.csv")