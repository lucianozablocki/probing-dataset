import pandas as pd
df=pd.read_csv("./rna_pdb_dataset.csv")

results=[]
seen_seqs=set()
removed_pdbids=[]
for index, row in df.iterrows():
    seq=row['sequence']
    if seq in seen_seqs:
        print(f"removing duplicate seq with id {row['pdbid']}", flush=True)
        removed_pdbids.append((row['pdbid'], "duplicate"))
        continue
    elif len(seq)<20 or len(seq)>500 or '&' in seq:
        print(f"removing seq with id {row['pdbid']} due to length {len(seq)} or invalid character &", flush=True)
        removed_pdbids.append((row['pdbid'], "invalid"))
    elif len(set(seq))<2:
        print(f"removing seq with id {row['pdbid']} due to repeated nucleotides: {seq}", flush=True)
        removed_pdbids.append((row['pdbid'], "repeated nucleotides"))
    else:
        print(f"{row['pdbid']},{seq}", flush=True)
        results.append(row)
    seen_seqs.add(seq)

results_df=pd.DataFrame(results)
results_df.to_csv("./sanitized_rnapdbdataset.csv", index=False)

removed_df=pd.DataFrame(removed_pdbids, columns=['pdbid', 'reason'])
removed_df.to_csv("./removed_rnapdbdataset.csv", index=False)