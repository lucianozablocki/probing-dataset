import json
import os
import pandas as pd

# updated alignment columns: pdb_id,rnagym_id,reactivity,reactivity_errors,alignment_seqA,alignment_seqB,experiment

# chain csv columns: chain,pdb_id,rnagym_id,experiment

# output columns: 'pdb_id'* 'rnagym_id'* 'experiment'* 'sequence' 'base_pairs' 'aligned_rnagym_seq'* 'aligned_pdb_seq'* 'reactivity'* 'reactivity_errors'* 'score' 'chain'

# sequence -> struct_df
# base_pairs -> struct_df
# score -> rnagym_seqs df
# chain -> chain_df

rnagym_alignments_complete_df = pd.read_parquet("https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/rnagym_vs_rnapdb_alignments_postprocessed.parquet")
struct_df = pd.read_csv("https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/rna_pdb_dataset_bp.csv")
chain_df = pd.read_csv("alignments_with_chain.csv")
folder_path='rnaglib_rnapdbee_diff/updated_aligments'

results=[]
import ast
# print(type(rnagym_alignments_complete_df["reactivity"].iloc[0]))
# rnagym_alignments_complete_df["reactivity"] = rnagym_alignments_complete_df["reactivity"].apply(ast.literal_eval)
# rnagym_alignments_complete_df["reactivity"].apply(
    # lambda x: [float(v) for v in ast.literal_eval(x)]
# )

# scan folder in which csv files are present
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    alignments = pd.read_csv(item_path)
    print(f"there are {len(alignments)} rows for file {item_path}")
    for idx, alignment in alignments.iterrows():
        pdb_id = alignment['pdb_id']
        rnagym_id = alignment['rnagym_id']
        experiment = alignment['experiment']
        aligned_rnagym_seq = alignment['alignment_seqA']
        aligned_pdb_seq = alignment['alignment_seqB']

        chain = chain_df[
            (chain_df['pdb_id']==pdb_id) &
            (chain_df['rnagym_id']==rnagym_id) &
            (chain_df['experiment']==experiment)]['chain'].values
        if len(chain) > 1:
            raise Exception("fatal")
        # print(chain)
        chain = chain[0]
        # print(chain)

        rnagym_alignment = rnagym_alignments_complete_df[
            (rnagym_alignments_complete_df['pdb_id']==pdb_id) &
            (rnagym_alignments_complete_df['sequence_id']==rnagym_id) &
            (rnagym_alignments_complete_df['experiment_type']==experiment)
        ]
        # print(rnagym_alignment.columns)
        # print(type(rnagym_alignment))

        # a combination of <pdb_id, rnagym_id, experiment> is not a unique alignment
        # but we are taking the first appearance elsewhere, so do the same here
        score = rnagym_alignment['local_alignment_score_bymin'].values[0]
        reactivity = rnagym_alignment['reactivity'].values[0]
        # print(type(reactivity))
        # print(json.dumps(reactivity))
        # break
        reactivity_errors = rnagym_alignment['reactivity_errors'].values[0]

        # print(type(rnagym_alignment))
        # score = rnagym_alignment['local_alignment_score_bymin']

        struct_row = struct_df[
            (struct_df['id'] == pdb_id) &
            (struct_df['chain'] == chain)]
        if len(struct_row) > 1:
            raise Exception("fatal")
        # print(type(struct_row))
        # struct_row = struct_row[0]
        # print(type(struct_row))
        sequence = struct_row['sequence']
        base_pairs = struct_row['base_pairs']

        updated = {}

        updated['pdb_id'] = pdb_id
        updated['rnagym_id'] = rnagym_id
        updated['aligned_rnagym_seq'] = aligned_rnagym_seq
        updated['aligned_pdb_seq'] = aligned_pdb_seq
        updated['reactivity'] = reactivity
        updated['reactivity_errors'] = reactivity_errors
        updated['experiment'] = experiment

        updated['chain'] = chain

        updated['score'] = score

        updated['sequence'] = sequence.values[0]
        updated['base_pairs'] = base_pairs.values[0]

        results.append(updated)

# write results to csv
out_df = pd.DataFrame(results)
out_df['reactivity'] = out_df['reactivity'].apply(lambda x: json.dumps(x.tolist()))
out_df['reactivity_errors'] = out_df['reactivity_errors'].apply(lambda x: json.dumps(x.tolist()))
out_df.to_csv("tool_mismatch.csv", index=False)
print(f"Written {len(out_df)} rows to tool_mismatch.csv")