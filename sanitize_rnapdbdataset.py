import pandas as pd
df=pd.read_csv("./TORNADO.csv")

results=[]
seen_seqs=set()
removed_pdbids=[]
for index, row in df.iterrows():
    seq=row['sequence'].lower()
    if seq in seen_seqs:
        print(f"removing duplicate seq with id {row['id']}", flush=True)
        removed_pdbids.append((row['id'], "duplicate"))
    elif len(seq)<20 or len(seq)>500 or '&' in seq:
        print(f"removing seq with id {row['id']} due to length {len(seq)} or invalid character &", flush=True)
        removed_pdbids.append((row['id'], "invalid"))
    elif len(set(seq))<2:
        print(f"removing seq with id {row['id']} due to repeated nucleotides: {seq}", flush=True)
        removed_pdbids.append((row['id'], "repeated nucleotides"))
    # check that sequence only contains A,C,G,U
    elif any(c not in 'acgu' for c in seq):
        print(f"removing seq with id {row['id']} due to invalid characters in sequence: {seq}", flush=True)
        removed_pdbids.append((row['id'], "invalid characters"))
    else:
        # print(f"{row['id']},{seq}", flush=True)
        row['sequence']=seq.upper()
        results.append(row)
    seen_seqs.add(seq)

results_df=pd.DataFrame(results)
results_df.to_csv("./sanitized_TORNADO.csv", index=False)

removed_df=pd.DataFrame(removed_pdbids, columns=['id', 'reason'])
removed_df.to_csv("./removed_TORNADO.csv", index=False)