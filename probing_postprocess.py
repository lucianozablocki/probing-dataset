import pandas as pd
# matches=pd.read_csv("./train_vs_rnapdb_matches_alignment1M114K.csv")# print(filtered['alignment_seqB'][0])
# filtered=matches[matches['local_alignment_score_bymin']>0.8].reset_index(drop=True)
# PDB_ID='5u3g'
# rnagym_ids=[]
# for idx, row in filtered.iterrows():
#     if row['pdb_id'] == PDB_ID:
#     # pdb_ids.append(row['pdb_id'])
#         rnagym_ids.append(row['train_sequence_id'])
#     # if row['local_alignment_score_bymin'] < 0.6:
#     #         print(f"error at row {row}")
# print(f"found {len(rnagym_ids)} sequences for PDB ID {PDB_ID} in alignment file")
# # print(len(pdb_ids))
# # from collections import Counter
# # pdb_id_counts=Counter(pdb_ids)
# # for pdb_id, count in pdb_id_counts.items():
# #     print(f"PDB ID: {pdb_id}, Count: {count}")

# filename='./train_data.csv'
# # id_to_search=''
# results=[]
# with pd.read_csv(filename, chunksize=100000) as reader:
#     for idx, chunk in enumerate(reader):
#         print(f"Processing chunk {idx}")
#         for idx2, row2 in chunk.iterrows():
#             if row2['sequence_id'] in rnagym_ids:
#                 results.append(row2)
#                 # print(f"Chunk {idx}, Row {idx2}: Sequence ID {row2['sequence_id']}")

# results_df=pd.DataFrame(results)
# # results_df.to_csv("./train_data_matching_4prf.csv",index=False)
# print(f"Total sequences for 4prf found: {len(results_df)}")

df=pd.read_csv('./rnagym_vs_rnapdb_alignments_extended.csv')
reactivity_cols=[]
reactivity_error_cols=[]
for number in range(1,207):
    # print(f"{number:04d}")
    reactivity_cols.append(f'reactivity_{number:04d}')
    reactivity_error_cols.append(f'reactivity_error_{number:04d}')

results=[]
for idx, row in df.iterrows():
    # handle NaN values
    for col in reactivity_cols + reactivity_error_cols:
        if pd.isna(row[col]):
            row[col] = -1000
    reactivity=row[reactivity_cols].values.tolist()
    reactivity_errors=row[reactivity_error_cols].values.tolist()
    for col in reactivity_cols + reactivity_error_cols:
        row.drop(col, inplace=True)
    # Example processing: print the first 5 reactivities and errors
    # print(f"Row {idx} Reactivities (first 5): {reactivities[50:55]}")
    # print(f"Row {idx} Reactivity Errors (first 5): {reactivity_errors[:5]}")
    result_row=row.to_dict()
    result_row['reactivity']=reactivity
    result_row['reactivity_errors']=reactivity_errors
    # result_row['pdb_id']=alignment_row['pdb_id']
    results.append(result_row)

results_df=pd.DataFrame(results)
results_df.to_parquet(f"./rnagym_vs_rnapdb_alignments_postprocessed.parquet",index=False)
print(f"Post-processed data saved. Total sequences processed: {len(results_df)}")