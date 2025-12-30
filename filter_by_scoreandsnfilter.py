import pandas as pd

matches=pd.read_csv("./rnagym_vs_rnapdb_alignments.csv")# print(filtered['alignment_seqB'][0])
filtered=matches[matches['local_alignment_score_bymin']>0.8].reset_index(drop=True)

rnagym_to_pdb_ids={}
# print(len(rnagym_ids))

# pdb_ids=[]
for idx, row in filtered.iterrows():
    # if row['pdb_id'] == '5u3g':
    rnagym_to_pdb_ids.setdefault(row['train_sequence_id'], []).append(row)
    # pdb_ids.append(row['pdb_id'])

print(len(rnagym_to_pdb_ids))

# from collections import Counter
# pdb_id_counts=Counter(pdb_ids)
# for pdb_id, count in pdb_id_counts.items():
#     print(f"PDB ID: {pdb_id}, Count: {count}")

filename='./train_data.csv'
# id_to_search=''
results=[]
flag=False
with pd.read_csv(filename, chunksize=100000) as reader:
    for idx, chunk in enumerate(reader):
        # if flag:
        #     break
        print(f"Processing chunk {idx}")
        for idx2, row2 in chunk.iterrows():
            # if flag:
            #     break
            if row2['sequence_id'] in rnagym_to_pdb_ids and row2['SN_filter']==1:
                # results.append(row2,**row)
                print("found sequence id with SN filter 1")
                for alignment_row in rnagym_to_pdb_ids[row2['sequence_id']]:
                    result_row=row2.to_dict()
                    result_row.update(alignment_row)
                    # result_row['pdb_id']=alignment_row['pdb_id']
                    results.append(result_row)
                    # flag=True
                    # break
                print(f"Chunk {idx}, Row {idx2}: Sequence ID {row2['sequence_id']}")

results_df=pd.DataFrame(results)
results_df.to_csv("./rnagym_vs_rnapdb_alignments_extended.csv",index=False)
print(f"Total sequences found: {len(results_df)}")