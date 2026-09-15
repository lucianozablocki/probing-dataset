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

RNG_SEED = 3
if RNG_SEED is not None:
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

structure_and_probing_df = pd.read_csv('../structure_and_probing.csv')
# synth_probing_dict = {
#     "pdb_id":
# }
synth_probing_list=[]
nt_weights={
    "DMS_MaP": [0.71, 0.29, 0.0, 0.0],
    "2A3_MaP": [0.36, 0.09, 0.2, 0.35],
}
# w_of_nt_2A3_MaP=
# w_of_nt_DMS_MaP=

DMS_MaP_A_values=[0.3363, 1.1369, 1.9376, 2.7382, 3.5388, 4.3394, 5.1401, 5.9407, 6.7413, 8.3426, 9.9438, 10.7444, 13.1463, 13.9469, 14.7476, 15.5482, 16.3488, 17.1494, 17.9501, 18.7507, 19.5513, 26.7569, 31.5607]
DMS_MaP_A_weights=[0.1829, 0.3934, 0.1334, 0.1087, 0.0536, 0.0289, 0.0096, 0.0165, 0.011, 0.0083, 0.0096, 0.0096, 0.0014, 0.0014, 0.0028, 0.0083, 0.0069, 0.0014, 0.0028, 0.0014, 0.0055, 0.0014, 0.0014]
DMS_MaP_C_values=[-0.5628, -0.2924, -0.0219, 0.2485, 0.5189, 0.7893, 1.0598, 1.3302, 1.6006, 1.871, 2.1415, 2.4119, 2.6823, 2.9527, 3.2232, 3.4936, 3.764, 4.0344, 4.3049, 4.5753, 4.8457, 5.1161, 5.3866, 5.657, 5.9274, 6.1978, 8.6317, 9.4429, 9.9838]
DMS_MaP_C_weights=[0.0034, 0.0034, 0.0448, 0.1, 0.0724, 0.0655, 0.1448, 0.069, 0.0586, 0.0448, 0.0862, 0.0517, 0.031, 0.031, 0.0138, 0.0103, 0.0103, 0.0276, 0.0414, 0.0448, 0.0069, 0.0034, 0.0103, 0.0034, 0.0034, 0.0034, 0.0069, 0.0034, 0.0034]
DMS_MaP_G_values=[]
DMS_MaP_G_weights=[]
DMS_MaP_U_values=[]
DMS_MaP_U_weights=[]

twoA3_MaP_A_values=[-0.0851, 0.0666, 0.2183, 0.37, 0.5218, 0.6735, 0.8252, 0.9769, 1.1287, 1.2804, 1.4321, 1.5838, 1.7356, 1.8873, 2.039, 2.1907, 2.3425, 2.4942, 2.6459, 2.7976, 2.9494, 3.1011, 3.2528, 3.4045, 3.708, 3.8597, 4.0114, 4.1632, 4.3149, 4.4666, 4.6183, 4.7701, 5.2252, 5.8321]
twoA3_MaP_A_weights=[0.0166, 0.1161, 0.0616, 0.0498, 0.0829, 0.0782, 0.0995, 0.0521, 0.0498, 0.0569, 0.0569, 0.0355, 0.0213, 0.019, 0.0118, 0.019, 0.0118, 0.0166, 0.0047, 0.0047, 0.0118, 0.019, 0.0071, 0.0047, 0.0237, 0.0047, 0.0071, 0.0095, 0.0024, 0.0024, 0.0142, 0.019, 0.0071, 0.0024]
twoA3_MaP_C_values=[-0.1618, -0.0034, 0.1551, 0.3135, 0.4719, 0.6303, 0.7888, 0.9472, 1.1056, 1.264, 1.4225, 1.5809, 1.7393, 1.8977, 2.0562, 3.6404, 5.0662, 5.6999, 6.0168]
twoA3_MaP_C_weights=[0.0096, 0.125, 0.1731, 0.1538, 0.1346, 0.0385, 0.0481, 0.0192, 0.0865, 0.0481, 0.0192, 0.0385, 0.0096, 0.0192, 0.0096, 0.0192, 0.0192, 0.0096, 0.0192]
twoA3_MaP_G_values=[-0.1539, 0.0143, 0.1824, 0.3506, 0.5188, 0.687, 0.8551, 1.0233, 1.1915, 1.3597, 1.5278, 1.696, 1.8642, 2.0324, 2.2005, 2.3687, 2.5369, 2.7051, 2.8732, 3.0414, 3.7141, 4.0505, 4.7232, 6.4049]
twoA3_MaP_G_weights=[0.0128, 0.0383, 0.0979, 0.1277, 0.0723, 0.1064, 0.0766, 0.1404, 0.0596, 0.0383, 0.0511, 0.0383, 0.017, 0.0043, 0.0255, 0.017, 0.0043, 0.0043, 0.0213, 0.017, 0.0043, 0.0085, 0.0128, 0.0043]
twoA3_MaP_U_values=[0.0595, 0.2285, 0.3975, 0.5665, 0.7355, 0.9045, 1.0735, 1.2425, 1.4115, 1.5805, 1.7495, 1.9185, 2.0875, 2.2565, 2.4255, 2.5945, 2.7635, 2.9325, 3.1015, 3.2705, 3.4395, 3.6085, 3.7775, 3.9465, 4.1155, 4.2845, 4.4535, 4.6225, 4.7915, 4.9605, 5.1295, 5.2985, 5.6365, 5.9745, 6.3125, 6.4815, 6.6505]
twoA3_MaP_U_weights=[0.0578, 0.053, 0.0602, 0.0723, 0.0482, 0.0289, 0.0795, 0.0771, 0.0313, 0.0554, 0.0337, 0.0289, 0.0241, 0.0217, 0.0193, 0.053, 0.0434, 0.0241, 0.012, 0.0145, 0.0313, 0.0169, 0.0169, 0.0193, 0.0048, 0.0024, 0.0096, 0.0024, 0.0096, 0.012, 0.0048, 0.012, 0.0024, 0.0024, 0.0048, 0.0024, 0.0072]
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


def low_pass_filter(signal, kernel=None):
    """Apply a simple low-pass filter (weighted moving average)."""
    if kernel is None:
        kernel = np.array([1.0, 2.0, 1.0], dtype=float)
    kernel = kernel / kernel.sum()
    signal_np = np.asarray(signal, dtype=float)
    return np.convolve(signal_np, kernel, mode='same')


def add_gaussian_noise(signal, mean=0.0, std=1.0):
    """Add Gaussian noise to a signal."""
    signal_np = np.asarray(signal, dtype=float)
    noise = np.random.normal(loc=mean, scale=std, size=signal_np.shape)
    return signal_np + noise

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
            # print(f"{chosen_nt} not found")
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

        # Post-process each synthetic signal with smoothing + Gaussian noise.
        synth_dms = add_gaussian_noise(low_pass_filter(synth_dms), mean=0.0, std=0.1).tolist()
        synth_2a3 = add_gaussian_noise(low_pass_filter(synth_2a3), mean=0.0, std=0.1).tolist()

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
df.to_csv('synthetic_probing_noised_seed3_std.1_updateddataset.csv', index=False)
