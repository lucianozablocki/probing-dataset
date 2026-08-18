from ast import literal_eval
import bisect
import json
import math
import operator
from pathlib import Path
import pandas as pd
import random
import numpy as np
import time
structure_and_probing_df = pd.read_csv('../structure_and_probing.csv')
# synth_probing_dict = {
#     "pdb_id":
# }
synth_probing_list=[]
nt_weights={
    "DMS_MaP": [0.7, 0.3, 0.0, 0.0],
    "2A3_MaP": [0.37, 0.09, 0.22, 0.32],
}
# w_of_nt_2A3_MaP=
# w_of_nt_DMS_MaP=

DMS_MaP_A_values=[0.3363, 1.1369, 1.9376, 2.7382, 3.5388, 4.3394, 5.1401, 5.9407, 6.7413, 9.9438, 10.7444, 13.1463, 14.7476, 15.5482, 16.3488, 17.1494, 19.5513, 26.7569, 31.5607]
DMS_MaP_A_weights=[0.1659, 0.431, 0.1422, 0.0884, 0.0431, 0.0302, 0.0086, 0.0237, 0.0151, 0.0108, 0.0129, 0.0022, 0.0022, 0.0065, 0.0065, 0.0022, 0.0043, 0.0022, 0.0022]
DMS_MaP_C_values=[-0.0947, 0.1639, 0.4224, 0.681, 0.9396, 1.1982, 1.4567, 1.7153, 1.9739, 2.2325, 2.491, 2.7496, 3.0082, 3.2668, 3.5253, 3.7839, 4.0425, 4.3011, 4.5596, 4.8182, 5.0768, 5.3354, 6.3697, 8.4383, 9.4726, 9.9897]
DMS_MaP_C_weights=[0.01, 0.0995, 0.0746, 0.0547, 0.0846, 0.1542, 0.0547, 0.0796, 0.0199, 0.0796, 0.0348, 0.0199, 0.0299, 0.0149, 0.01, 0.01, 0.0299, 0.0448, 0.0597, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
DMS_MaP_G_values=[]
DMS_MaP_G_weights=[]
DMS_MaP_U_values=[]
DMS_MaP_U_weights=[]

twoA3_MaP_A_values=[0.07, 0.2, 0.33, 0.46, 0.6, 0.73, 0.86, 0.99, 1.12, 1.26, 1.39, 1.52, 1.65, 1.79, 1.92, 2.05, 2.18, 2.32, 2.45, 2.58, 2.71, 2.84, 2.98, 3.11, 3.24, 3.37, 3.51, 3.64, 3.77, 4.04, 4.17, 4.3, 4.57, 4.7, 4.83, 5.23]
twoA3_MaP_A_weights=[0.08, 0.05, 0.03, 0.09, 0.07, 0.05, 0.07, 0.05, 0.06, 0.05, 0.04, 0.05, 0.03, 0.02, 0.01, 0.01, 0.02, 0.02, 0.02, 0.0, 0.01, 0.02, 0.01, 0.02, 0.01, 0.0, 0.0, 0.01, 0.02, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01]
twoA3_MaP_C_values=[0.02, 0.17, 0.33, 0.48, 0.63, 0.79, 0.94, 1.1, 1.25, 1.4, 1.56, 1.71, 1.86, 2.02, 3.71, 5.1, 6.02]
twoA3_MaP_C_weights=[0.1, 0.21, 0.15, 0.13, 0.03, 0.06, 0.01, 0.09, 0.04, 0.04, 0.03, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01]
twoA3_MaP_G_values=[-0.17, -0.05, 0.08, 0.2, 0.32, 0.45, 0.57, 0.69, 0.82, 0.94, 1.07, 1.19, 1.31, 1.44, 1.56, 1.68, 1.81, 1.93, 2.05, 2.18, 2.3, 2.42, 2.55, 2.79, 2.92, 3.04, 4.15, 4.65]
twoA3_MaP_G_weights=[0.01, 0.02, 0.05, 0.07, 0.04, 0.11, 0.1, 0.06, 0.06, 0.08, 0.1, 0.04, 0.02, 0.03, 0.06, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02]
twoA3_MaP_U_values=[0.11, 0.28, 0.45, 0.62, 0.78, 0.95, 1.12, 1.29, 1.45, 1.62, 1.79, 1.96, 2.13, 2.29, 2.46, 2.63, 2.8, 2.96, 3.13, 3.3, 3.47, 3.63, 3.8, 3.97, 4.14, 4.3, 4.47, 4.64, 4.81, 4.97, 5.14, 5.31, 5.65, 6.32, 6.48, 6.65]
twoA3_MaP_U_weights=[0.05, 0.08, 0.04, 0.05, 0.03, 0.06, 0.06, 0.08, 0.04, 0.07, 0.03, 0.03, 0.01, 0.02, 0.05, 0.06, 0.03, 0.01, 0.02, 0.02, 0.02, 0.03, 0.01, 0.02, 0.0, 0.0, 0.01, 0.0, 0.0, 0.02, 0.01, 0.02, 0.0, 0.01, 0.0, 0.01]

nts_probing_values_and_weights_by_experiment={
    "DMS_MaP": {
        "A": {
            "values": DMS_MaP_A_values,
            "weights": DMS_MaP_A_weights,
        },
        "C": {
            "values": DMS_MaP_C_values,
            "weights": DMS_MaP_C_weights,
        },
        "G": {
            "values": DMS_MaP_G_values,
            "weights": DMS_MaP_G_weights,
        },
        "U": {
            "values": DMS_MaP_U_values,
            "weights": DMS_MaP_U_weights,
        },
    },
    "2A3_MaP": {
        "A": {
            "values": twoA3_MaP_A_values,
            "weights": twoA3_MaP_A_weights,
        },
        "C": {
            "values": twoA3_MaP_C_values,
            "weights": twoA3_MaP_C_weights,
        },
        "G": {
            "values": twoA3_MaP_G_values,
            "weights": twoA3_MaP_G_weights,
        },
        "U": {
            "values": twoA3_MaP_U_values,
            "weights": twoA3_MaP_U_weights,
        },
    }
}

def get_synth_probing_value(nt, experiment):
    nts_probing_values_and_weights=nts_probing_values_and_weights_by_experiment[experiment]
    values_and_weights=nts_probing_values_and_weights[nt]
    values=values_and_weights['values']
    weights=values_and_weights['weights']
    # print(values_and_weights)
    return random.choices(values, weights=weights, k=1)[0]

def sort_synth_probing_values_by_nt(sequence, region, experiment, n):
    positions=[]
    synth_probing_list=[]
    # print(f"generating {n} maxs for region {sequence[region[0]:region[-1]+1]} ({region[0]}-{region[-1]}) for experiment {experiment}")
    if len(region)>1:
        chosen_nt=random.choices(['A', 'C', 'G', 'U'], weights=nt_weights[experiment],k=1)[0]
        # print(chosen_nt)
        # for chosen_nt in chosen_nts:
        #     positions=[i for i, nt in enumerate(sequence[region[0]:region[-1]+1]) if nt == chosen_nt]
        #     chosen_position=random.sample(positions, 1)[0]
        #     synth_probing_value=get_synth_probing_value(chosen_nt, experiment)
        #     synth[chosen_position + region[0]] = synth_probing_value
        # print(sequence[region[0]:region[-1]+1])
        positions=[i for i, nt in enumerate(sequence[region[0]:region[-1]+1]) if nt == chosen_nt]
        # print("positions", positions)
        # sequence[region[0]:region[-1]+1].find(nt)
        while not positions:
            print(f"{chosen_nt} not found")
            chosen_nt=random.choices(['A', 'C', 'G', 'U'], weights=nt_weights[experiment],k=1)[0]
            # print("chosen_nt", chosen_nt)
            positions=[i for i, nt in enumerate(sequence[region[0]:region[-1]+1]) if nt == chosen_nt]
            # print(positions)
        # take n elements randomly from positions
        # print(len(positions))
        # print(positions)
        if len(positions)>=n:
            positions=random.sample(positions, n)
            # print("sampled positions", positions)
            for i, pos in enumerate(positions):
                positions[i] = pos + region[0]
                # print(sequence[positions[i]])
                assert(positions[i] in region)
                synth_probing_value=get_synth_probing_value(chosen_nt, experiment)
                synth_probing_list.append(synth_probing_value)
                # print(len(synth_probing_list))
        else:
            for i, pos in enumerate(positions):
                positions[i] = pos + region[0]
                assert(positions[i] in region)
                synth_probing_value=get_synth_probing_value(chosen_nt, experiment)
                synth_probing_list.append(synth_probing_value)
    else:
        # print("short region")
        positions=[region[0]]
        chosen_nt=sequence[positions[0]]
        # print(pos)
        # print(nt)
        synth_probing_list=[get_synth_probing_value(chosen_nt, experiment)]
        # print(synth_probing_value)
    return positions, synth_probing_list


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
unknown_experiment_count=0

grouped_df=structure_and_probing_df.groupby(['pdb_id','chain'])
# print(len(grouped_df))
total_processed_rows=0
N=5 # we will generate as many max values as ceil(len(region)/N)
synth_by_seq=5
synth_probing_list=[]
for (pdb_id, chain), group in grouped_df:
    unpaired_regions = []
    # if pdb_id!='6zmo' or chain!='CB':
    #     continue
    # if pdb_id!='1duh' or chain!='A':
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
    print(unpaired_regions)
    for _ in range(synth_by_seq):
        synth_dms = [0]*len(sequence)
        synth_2a3 = [0]*len(sequence)
        for r in unpaired_regions:
            # print(r)
            region_len = len(r)
            n_maxs_to_gen = math.ceil(region_len/N)
            # print("dms")
            if (not sequence[r[0]:r[-1]+1].find('A')==-1 or not sequence[r[0]:r[-1]+1].find('C')==-1):
                # for _ in range(n_maxs_to_find):
                positions, synth_probing_values = sort_synth_probing_values_by_nt(sequence, r, 'DMS_MaP', n_maxs_to_gen)
                for position, value in zip(positions, synth_probing_values):
                    synth_dms[position]=value

            # print("2a3")
            # for _ in range(n_maxs_to_find):
            positions, synth_probing_values = sort_synth_probing_values_by_nt(sequence, r, '2A3_MaP', n_maxs_to_gen)
            for position, value in zip(positions, synth_probing_values):
                synth_2a3[position]=value

        synth_probing_list.append({
            "pdb_id": pdb_id,
            "chain": chain,
            "reactivity": synth_dms,
            "experiment": 'DMS_MaP',
            "sequence": sequence,
            "dot_bracket": dot_bracket,
        })

        synth_probing_list.append({
            "pdb_id": pdb_id,
            "chain": chain,
            "reactivity": synth_2a3,
            "experiment": '2A3_MaP',
            "sequence": sequence,
            "dot_bracket": dot_bracket,
        })
print(f"generated {len(synth_probing_list)} synth signals")
df=pd.DataFrame(synth_probing_list)
df.to_csv('synthetic_probing.csv', index=False)
