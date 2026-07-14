from ast import literal_eval
import pandas as pd
import numpy as np

structure_and_probing_df = pd.read_csv('structure_and_probing.csv')

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
gap_pdbseq_count=0
gap_rnagymseq_count=0
len_mismatch_count=0
grouped_df=structure_and_probing_df.groupby(['pdb_id','chain'])
nts_at_which_max_occurs=[]
max_by_nt={'A': [], 'C': [], 'G': [], 'U': []}
total_processed_rows=0
N=5 # we will look for as many max values as floor(len(region)/N)
for (pdb_id, chain), group in grouped_df:
    unpaired_regions = []
    # print(f"Processing pdb_id={pdb_id}, chain={chain}")
    # if pdb_id!='5gah':
    #     continue
    print(f"Processing pdb_id={pdb_id}, chain={chain}")
    dot_bracket = group.iloc[0]['dot_bracket']
    sequence = group.iloc[0]['sequence']
    # print(sequence)
    row_as_dict=group.to_dict('records')
    # print(dot_bracket)
    # detect not paired regions by counting consecutive dots in the dot_bracket notation
    current_region = []
    # test_dot_bracket="...((..))..((..)).."
    for idx, char in enumerate(dot_bracket):
        if char == '.':
            current_region.append(idx)
        else:
            if current_region:
                unpaired_regions.append(current_region)
                current_region = []
    if current_region:
        unpaired_regions.append(current_region)
    max_by_region=[-float('inf')]*len(unpaired_regions)
    max_nt_by_region=[""]*len(unpaired_regions)
    # print(max_by_region)
    for row in row_as_dict:
        total_processed_rows+=1
        # print(f"row: {row}")
        start, end = find_alignment_bounds(row['aligned_pdb_seq'])
        # print(start)
        # print(end)
        # print("---------------")
        if not (len(dot_bracket) == len(sequence) == (end - start + 1)):
            len_mismatch_count+=1
            break
        if '-' in row['aligned_pdb_seq'][start:end+1]:
            # print(f"gap for pdb {row['pdb_id']}")
            gap_pdbseq_count+=1
            break
        if '-' in row['aligned_rnagym_seq'][start:end+1]:
            # print(f"gap for rnagym {row['pdb_id']}")
            gap_rnagymseq_count+=1
            break
        for idx, region in enumerate(unpaired_regions):
            region_len=len(region)
            probing_start = region[0] + start
            probing_end = region[-1] + start
            probing_values = literal_eval(row['reactivity'])[probing_start:probing_end+1]
            n_maxs_to_find=math.floor(region_len/N)
            indexed = list(enumerate(probing_values))
            top_n_maxs_to_find = sorted(indexed, key=operator.itemgetter(1))[-n_maxs_to_find:]
            indxs_and_values=list(reversed(top_n_maxs_to_find)) # list of n max (idx,val) tuples 
            # for _ in maximums_to_find:
            #     # probing_values_cpy=probing_values
            #     # Find the maximum value
            #     if max_idx is not None:
            #         max_val = max(probing_values[:max_idx]+probing_values[max_idx+1:])
            #         # Find the first index of that value
            #         max_idx = probing_values.index(max_val)
            #     else:
            #         max_val = max(probing_values[:max_idx]+probing_values[max_idx+1:])
            #         # Find the first index of that value
            #         max_idx = probing_values.index(max_val)

            nt_at_which_max_occurs=sequence[region[max_idx]]

            if max_val > max_by_region[idx] and max_val != -1000:
                aligned_rnagym_seq=row['aligned_rnagym_seq']
                aligned_pdb_seq=row['aligned_pdb_seq']
                rnagym_nt = aligned_rnagym_seq[region[max_idx]+start]
                if rnagym_nt == nt_at_which_max_occurs:
                    max_by_region[idx] = max_val
                    max_nt_by_region[idx] = nt_at_which_max_occurs
                else:
                    continue
    # if pdb_id=='1il2':
    #     print(f"unpaired_regions: {unpaired_regions}")
    #     print(f"max_by_region: {max_by_region}")
    #     print(f"max_nt_by_region: {max_nt_by_region}")
    nts_at_which_max_occurs.extend(max_nt_by_region)
    nts_at_which_max_occurs = [nt for nt in nts_at_which_max_occurs if nt != ""] # filter out empty chars
    # for nt_or_blank in nts_at_which_max_occurs:
    for nt, max_val in zip(max_nt_by_region, max_by_region):
        if nt in max_by_nt: # there could be an empty char if no max was found for a region
            max_by_nt[nt].append(max_val)

print(len_mismatch_count)
print(gap_pdbseq_count)
print(gap_rnagymseq_count)
print("\n")
print(max_by_nt)
print(nts_at_which_max_occurs)
print(f"Total processed rows: {total_processed_rows}")
