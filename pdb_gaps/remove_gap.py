import json
import pandas as pd

PARQUET_URL = (
    "https://raw.githubusercontent.com/lucianozablocki/probing-dataset"
    "/refs/heads/main/rnagym_vs_rnapdb_alignments_postprocessed.parquet"
)

# # updated_aligments_multichain=
df = pd.read_parquet(PARQUET_URL)

struct_df = pd.read_csv("https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/rna_pdb_dataset_bp.csv")

chain_df = pd.read_csv("alignments_with_chain.csv")
# print(df.columns)
SEQB_GAPS=['7mlx', '1mms', '5nwq', '5d8h', '5gah', '1il2', '5lzs', '6r5q', '5lys', '6pmo', '3r4f', '3npq', '8am9', '6zym', '8r6c', '8d9k', '5el4', '5ml7', '1l9a', '7p6z', '3kfu', '8s1p', '6mj0', '5ib8', '7d6z', '6wzr', '7mdl', '8k1e']
UPDATED_ALIGNMENTS_TOOLDIFF=["1e8o", "4jf2", "4mgn", "4mgn", "4mgn", "5aox", "5axm", "5d5l", "6prv", "6xko", "7k16", "7n2v", "8peg", "8v1i", "8v1i", "3k1v", "4mgn", "5aox", "5ns3", "7d8o", "7mky", "8g9z",]

# alignments_with_chain = df['pdb_id'].unique()

# print(alignments_with_chain)
# UPDATED_ALIGNMENTS_TOOLDIFF.extend(alignments_with_chain)
def find_alignment_bounds(alignment_seqB):
    """Find start and end of seqB in the alignment (first/last non-gap positions)."""
    start = None
    end = None
    for idx, nuc in enumerate(alignment_seqB):
        if nuc != '-':
            if start is None:
                start = idx
            end = idx
    return start, end

results = []
count=[]
seen=[]
for pdb_id_with_gap in SEQB_GAPS:
    print(pdb_id_with_gap)
    count_for_pdb_id=0
    if pdb_id_with_gap in UPDATED_ALIGNMENTS_TOOLDIFF:
        print(f"pdb {pdb_id_with_gap} alignment was modified, skipping")
        continue
    rows = df[df['pdb_id'] == pdb_id_with_gap]

    for idx, row in rows.iterrows():
        if (row['pdb_id'], row['sequence_id'], row['experiment_type']) in seen:
            continue
        seen.append((row['pdb_id'], row['sequence_id'], row['experiment_type']))
        chain = chain_df[(chain_df['pdb_id']==pdb_id_with_gap) & (chain_df['rnagym_id']==row['sequence_id']) & (chain_df['experiment']==row['experiment_type'])]['chain'].values
        if len(chain)==1:
            chain = chain[0]
        else:
            raise Exception("big error") # because a combination of <pdb_id,rnagym_id,experiment_type> is unique in alignments_with_chain.csv
        struct_df_rows = struct_df[(struct_df['id'] == pdb_id_with_gap) & (struct_df['chain'] == chain)]
        # if len(struct_df_rows) >1 :
        #     print(f"more than one chain for pdb {pdb_id_with_gap}, mark it first")
        #     continue
        pdb_seq = row['alignment_seqB']
        rnagym_seq = row['alignment_seqA']
        start, end = find_alignment_bounds(pdb_seq)

        dash_pos = [pos for pos, char in enumerate(pdb_seq) if char == '-']
        inside_alignment_pos = [pos for pos in dash_pos if start < pos < end]
        # print(rnagym_seq)
        # print(pdb_seq)

        reactivity = list(row.get('reactivity', []))
        reactivity_errors = list(row.get('reactivity_errors', []))

        if not inside_alignment_pos:
            updated = {}

            updated['pdb_id'] = row['pdb_id']
            updated['rnagym_id'] = row['sequence_id']
            updated['sequence'] = struct_df_rows.iloc[0]['sequence']
            updated['base_pairs'] = struct_df_rows.iloc[0]['base_pairs']

            updated['aligned_rnagym_seq'] = rnagym_seq
            updated['aligned_pdb_seq'] = pdb_seq

            updated['reactivity'] = reactivity
            updated['reactivity_errors'] = reactivity_errors
            updated['score'] = row['local_alignment_score_bymin']
            updated['experiment'] = row['experiment_type']
            updated['chain'] = chain

            results.append(updated)
            # results.append(row.to_dict())
            # raise Exception(f"no gaps inside alignment for pdb {pdb_id_with_gap}, this is an issue")
            count_for_pdb_id+=1
            continue


        # Remove columns in reverse order so earlier deletions don't shift later indices.
        for col in sorted(inside_alignment_pos, reverse=True):
            if rnagym_seq[col] != '-':
                rnagym_seq_index = col - rnagym_seq[:col].count('-')
                reactivity = reactivity[:rnagym_seq_index] + reactivity[rnagym_seq_index + 1:]
                reactivity_errors = reactivity_errors[:rnagym_seq_index] + reactivity_errors[rnagym_seq_index + 1:]
            rnagym_seq = rnagym_seq[:col] + rnagym_seq[col + 1:]
            pdb_seq = pdb_seq[:col] + pdb_seq[col + 1:]

        # updated = row.to_dict()
        updated = {}

        updated['pdb_id'] = row['pdb_id']
        updated['rnagym_id'] = row['sequence_id']
        updated['sequence'] = struct_df_rows.iloc[0]['sequence']
        updated['base_pairs'] = struct_df_rows.iloc[0]['base_pairs']

        updated['aligned_rnagym_seq'] = rnagym_seq
        updated['aligned_pdb_seq'] = pdb_seq

        updated['reactivity'] = reactivity
        updated['reactivity_errors'] = reactivity_errors
        updated['score'] = row['local_alignment_score_bymin']
        updated['chain'] = chain

        results.append(updated)
        count_for_pdb_id += 1

    print(f"processed {len(rows)} rows for {pdb_id_with_gap}")
    print(f"but only {count_for_pdb_id} after removing dupes")
    # count=0
    count.append(count_for_pdb_id)
    # print(count)
    # print(len(results))

print(len(results))
print(sum(count))
if results:
    out_df = pd.DataFrame(results)
    out_df['reactivity'] = out_df['reactivity'].apply(json.dumps)
    out_df['reactivity_errors'] = out_df['reactivity_errors'].apply(json.dumps)
    out_df.to_csv("alignments_seqb_gaps_removed.csv", index=False)
    print(f"Written {len(out_df)} rows to alignments_seqb_gaps_removed.csv")