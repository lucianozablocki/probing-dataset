import pandas as pd
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from Bio import pairwise2

Match = 1
Mismatch = -2
GapOpen=-5
GapExtend=-2

simbolos=["A","C","G","U"]

match_dic = {}
for simbolA in simbolos:
    for simbolB in simbolos:    
        if simbolA==simbolB:
            match_dic[(simbolA, simbolB)] = Match
        else:
            match_dic[(simbolA, simbolB)] = Mismatch

def normalize_sequence(raw: Optional[str]) -> str:
    """Normalize a sequence by stripping whitespace, upper-casing, and T→U."""

    if not isinstance(raw, str):
        return ""
    cleaned = ''.join(raw.split()).upper()
    return cleaned

filename_1="./train_data.csv"
filename_2="./rna_pdb_dataset.csv"
output_file="./train_vs_rnapdb_matches_alignment.csv"

# df_1 = pd.read_csv(filename_1)
df_2 = pd.read_csv(filename_2)

max_len_rnapdb=470 # max_len_global
results=[]
write_batch_size = 100  # Write results every 100 alignments
first_write = True

with pd.read_csv(filename_1, chunksize=10**3) as reader:
    print("reading")
    for idx, chunk in enumerate(reader): # paralelizar
        print(f"########## CHUNK NUMBER {idx} ##########")
        for idx2, record2 in df_2.iterrows():
            if idx2%10==0:
                print(f"processing {idx2} of {len(df_2)}")
            seqB=normalize_sequence(record2['sequence'])
            if '&' in seqB:
                print(f"skipping pdb seq with id: {record2['pdbid']} due to invalid character &")
                continue
            if len(seqB)>500 or len(seqB)<10:
                print(f"skipping pdb seq with id: {record2['pdbid']} due to length {len(seqB)}")
                continue
            for idx, record in chunk.iterrows():
                seqA=normalize_sequence(record['sequence'])
                # if seqA==seqB:
                #     print(f"Exact match found: {seqA}")
                #     results.append({
                #         "seqA": seqA,
                #         "seqB": seqB,
                #         "type": "exact",
                #         "train_sequence_id": record['sequence_id'],
                #         "dataset_name": record['dataset_name'],
                #         "signal_to_noise": record['signal_to_noise'],
                #         'pdb_id': record2['pdbid'],
                #     })
                # elif (seqA in seqB and len(seqA)>10) or (seqB in seqA and len(seqB)>10):
                #     print(f"Subsequence match found: {seqA} in {seqB} or {seqB} in {seqA}")
                #     results.append({
                #         "seqA": seqA,
                #         "seqB": seqB,
                #         "type": "subsequence",
                #         "train_sequence_id": record['sequence_id'],
                #         "dataset_name": record['dataset_name'],
                #         "signal_to_noise": record['signal_to_noise'],
                #         'pdb_id': record2['pdbid'],
                #     })
                alignm = pairwise2.align.localds(seqA, seqB, match_dic, GapOpen, GapExtend, one_alignment_only=True)[0]
                equal_nucleotides_count = sum(a==b for a,b in zip(alignm.seqA, alignm.seqB))
                max_len_local = max(len(alignm.seqB),len(alignm.seqA))
                local_IDscore_bymax = equal_nucleotides_count / max_len_local
                global_IDscore_bymax = equal_nucleotides_count / max_len_rnapdb
                min_len_local = min(len(alignm.seqB),len(alignm.seqA))
                local_IDscore_bymin = equal_nucleotides_count / min_len_local
                # print(f"between train seq id {record['sequence_id']} and pdb seq id {record2['pdbid']} found alignment with IDscore: {IDscore:.4f}")
                results.append({
                    "seqA": seqA,
                    "seqB": seqB,
                    "type": "alignment",
                    "train_sequence_id": record['sequence_id'],
                    "dataset_name": record['dataset_name'],
                    "signal_to_noise": record['signal_to_noise'],
                    'pdb_id': record2['pdbid'],
                    'global_alignment_score_bymax': global_IDscore_bymax,
                    'local_alignment_score_bymax': local_IDscore_bymax,
                    'local_alignment_score_bymin': local_IDscore_bymin,
                })
                
                # Write results incrementally
                if len(results) >= write_batch_size:
                    print(f"Writing batch of {len(results)} results to {output_file}...")
                    df_batch = pd.DataFrame(results)
                    if first_write:
                        df_batch.to_csv(output_file, index=False, mode='w')
                        first_write = False
                    else:
                        df_batch.to_csv(output_file, index=False, mode='a', header=False)
                    results = []  # Clear the results list

# Write any remaining results
if results:
    print(f"Writing final batch of {len(results)} results to {output_file}...")
    df_batch = pd.DataFrame(results)
    if first_write:
        df_batch.to_csv(output_file, index=False, mode='w')
    else:
        df_batch.to_csv(output_file, index=False, mode='a', header=False)

print(f"Finished writing all results to {output_file}")