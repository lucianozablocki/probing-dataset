import pandas as pd
import json
# output columns: 'pdb_id' 'rnagym_id' 'experiment' 'sequence' 'base_pairs' 'aligned_rnagym_seq' 'aligned_pdb_seq' 'reactivity' 'reactivity_errors' 'score' 'chain'

PLOTTED_PDBIDS=['4v4j', '6gz4', '6ah3', '7o7z', '2xd0', '1mji', '4lvw', '4pr6', '4prf', '3jbu', '3jbv', '4v6d', '7p3k', '5l3p', '8g7p', '3moj', '1fir', '5ccb', '7nwg', '4enc', '2a43', '4lck', '7yse', '4rmo', '6yl5', '8b2l', '5vt0', '6xh2', '3t4b', '7mrl', '5hr6', '6yal', '8cd1', '3d2v', '1et4', '1zci', '1duh', '3lqx', '7qr3', '7qr4', '2der', '5hr7', '6q9a', '1mfq', '3sux', '3wfs', '7o5b', '8h6l', '8k2z', '9dtt', '3gs5', '4p5j', '4xwf', '387d', '7vft', '5mwi', '6e8u', '4v83', '4xej', '5btp', '7kga', '1kh6', '6zmo', '3egz', '4znp', '7b5k', '7eqj', '3rg5', '7zjw', '6vwl', '7ot5']
TOOL_MISMATCH_BUT_NO_ALIGNMENT_MODIFIED=['9f9s', '1xjr', '4oqu', '1l2x', '6cu1', '5e81', '7n2v', '7zta', '8yup', '5ju8', '6fz0', '6mwn', '1y27', '8peg']

PLOTTED_PDBIDS.extend(TOOL_MISMATCH_BUT_NO_ALIGNMENT_MODIFIED)

rnagym_alignments_complete_df = pd.read_parquet("https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/rnagym_vs_rnapdb_alignments_postprocessed.parquet")
struct_df = pd.read_csv("https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/rna_pdb_dataset_bp.csv")
chain_df = pd.read_csv("alignments_with_chain.csv")

# for pdb_id in set(chain_df['pdb_id'].values):
#     chains = set(chain_df[chain_df['pdb_id']==pdb_id]['chain'])
#     # print(chains)
#     if len(chains) >1:
#         print(pdb_id)
#         print(len(chains))
seen=[]
results=[]
for pdb_id in PLOTTED_PDBIDS:
    rnagym_alignment_rows = rnagym_alignments_complete_df[rnagym_alignments_complete_df['pdb_id'] == pdb_id]
    for idx, rnagym_alignment in rnagym_alignment_rows.iterrows():
        if (rnagym_alignment['pdb_id'],
            rnagym_alignment['sequence_id'],
            rnagym_alignment['experiment_type']) in seen:
            # this is wrong as <pdb_id,rnagym_id,experiment> does not constitute a unique key
            # we are forgetting about CHAIN
            continue
        seen.append((rnagym_alignment['pdb_id'], rnagym_alignment['sequence_id'], rnagym_alignment['experiment_type']))
    
        pdb_seq = rnagym_alignment['alignment_seqB']
        rnagym_seq = rnagym_alignment['alignment_seqA']
        score = rnagym_alignment['local_alignment_score_bymin']
        reactivity = rnagym_alignment['reactivity']
        reactivity_errors = rnagym_alignment['reactivity_errors']
        experiment = rnagym_alignment['experiment_type']

        chain = chain_df[
            (chain_df['pdb_id']==pdb_id) &
            (chain_df['rnagym_id']==rnagym_alignment['sequence_id']) &
            (chain_df['experiment']==rnagym_alignment['experiment_type'])]['chain'].values
        if len(chain)==1:
            chain = chain[0]
        else:
            raise Exception("big error") # this wont happen as we are (wrongly) filtering repeated <pdb_id,rnagym_id,experiment>
        struct_df_rows = struct_df[(struct_df['id'] == pdb_id) & (struct_df['chain'] == chain)]

        updated = {}

        updated['pdb_id'] = pdb_id
        updated['rnagym_id'] = rnagym_alignment['sequence_id']
        updated['sequence'] = struct_df_rows.iloc[0]['sequence']
        updated['base_pairs'] = struct_df_rows.iloc[0]['base_pairs']

        updated['aligned_rnagym_seq'] = rnagym_seq
        updated['aligned_pdb_seq'] = pdb_seq

        updated['reactivity'] = reactivity
        updated['reactivity_errors'] = reactivity_errors
        updated['score'] = score
        updated['experiment'] = experiment
        updated['chain'] = chain

        results.append(updated)

print(len(results))
# print(sum(count))
out_df = pd.DataFrame(results)
out_df['reactivity'] = out_df['reactivity'].apply(lambda x: json.dumps(x.tolist()))
out_df['reactivity_errors'] = out_df['reactivity_errors'].apply(lambda x: json.dumps(x.tolist()))
out_df.to_csv("alignments_rows_completed.csv", index=False)
print(f"Written {len(out_df)} rows to aligment_rows_completed.csv")