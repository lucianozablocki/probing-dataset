import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,3)
rnapdb_dataset=pd.read_csv("https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/rna_pdb_dataset_bp.csv")
rnagym_seqs=pd.read_parquet("https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/rnagym_vs_rnapdb_alignments_postprocessed.parquet")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
BOLD = "\033[1m"
RESET = "\033[0m"

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
from Bio import pairwise2

def compute_alignment_bounds(probing, seqA, seqB):
  for idx, p in enumerate(probing):
    # print(p)
    # print(probing[idx+1])
    if idx==len(probing)-1:
      # print(f"probing values ends at {idx+1}")
      break
    if p==float(-1000) and probing[idx+1]!=float(-1000):
      # print(f"probing values starts at {idx+1}")
      probing_start=idx+1
    if p!=float(-1000) and probing[idx+1]==float(-1000):
      # print(f"probing values ends at {idx}")
      probing_end=idx

  alignm = pairwise2.align.localds(seqA, seqB, match_dic, GapOpen, GapExtend, one_alignment_only=True)[0]
  lens_array = [len(seqA), len(seqB)]
  min_pos = np.argmin(lens_array)
  min_len_local = lens_array[min_pos]
  equal_nucleotides_count = sum(a == b for a, b in zip(alignm.seqA, alignm.seqB))
  local_IDscore_bymin = equal_nucleotides_count / min_len_local
  # print(alignm.seqA)

  # print(len(r_list))
  # print(len(seqA))
  # print(len(alignm.seqA))
  not_start=True
  start_alignm = None
  end_alignm = None
  for idx, (nucleotideA, nucleotideB) in enumerate(zip(alignm.seqA,alignm.seqB)):
    # print(idx)
    if nucleotideA==nucleotideB:
      print(f"{BOLD}{nucleotideA}{RESET}",end="")
    else:
      print(f"{nucleotideA}",end="")

    # if nucleotideB!="-" and idx==0 and not_start:
    #   start_alignm=idx
    #   not_start=False
    # elif nucleotideB!="-" and idx==len(alignm.seqB)-1 and not_start==False:
    #   end_alignm=idx
    #   not_start=True
    # elif nucleotideB!="-" and not_start:
    #   start_alignm=idx
    #   not_start=False
    # elif nucleotideB=="-" and not_start==False:
    #   end_alignm=idx
    #   not_start=True

    if nucleotideB != "-":
      if not_start:
        start_alignm = idx
        not_start = False
      end_alignm = idx  # Always update end to the last non-gap position
  print("\n",end="")
  print(alignm.seqB)

  # print("-"*start_alignm,end="")

  # for idx, nucleotideB in enumerate(seqB):
  #   # if nucleotideB=='-':
  #   #   print("-",end="")
  #   # else:
  #   print(f"{idx%10}",end="")

  # print("-"*(len(alignm.seqA)-len(seqB)-start_alignm),end="")
  # print("\n")

  # print(f"alignment starts at {start_alignm}")
  # print(f"alignment ends at {end_alignm}")


  # if start_alignm < probing_start:
  #   # draw=False
  #   start_alignm=probing_start
  #   draw=True
  # elif end_alignm > probing_end:
  #   end_alignm=probing_end
  #   draw=True
  # else:
  #   draw=True

  return alignm, start_alignm, end_alignm, True

def plot_probings_and_pseudoprobing(pdb_id, rows_of_pdbid, start_alignm, end_alignm, early_cut):
  plt.figure()
  # print(len(probing))
  # print(len(struct))
  # print(type(struct))
  # rows=rnagym_seqs[rnagym_seqs['pdb_id']==pdb_id]
  plt.grid()
  struct=rnapdb_dataset[rnapdb_dataset['id']==pdb_id]['base_pairs'].values[0]
  pseudo_probing=[1 if s=="." else 0 for s in list(struct)]
  pseudo_probing=np.array(pseudo_probing)-1.5

  # Calculate match frequency at each position across all rows
  seq_len = end_alignm - start_alignm - early_cut
  ref_seq = rows_of_pdbid[0]['alignment_seqB'][start_alignm:end_alignm-early_cut]

  # Collect all seqA alignments and compute match frequency per position
  match_counts = [0] * seq_len
  total_seqs = len(rows_of_pdbid)
  for idx, row in enumerate(rows_of_pdbid):
    # print(row)
    seqA = row.get('alignment_seqA')#, alignm.seqA[start_alignm:end_alignm-early_cut])  # fallback to alignm param
    for i, (nB, nA) in enumerate(zip(ref_seq, seqA)):
      if i < seq_len and nA == nB and nA != '-':
        match_counts[i] += 1

  match_freq = [c / total_seqs if total_seqs > 0 else 0 for c in match_counts]

  x_tick_labels = [
    f"{nB}$_{{{i+1}}}$" if (i+1)%10==0 else f"{nB}"
    for i, nB in enumerate(ref_seq)
  ]

  ax = plt.gca()
  ax.set_xticks(range(seq_len))
  ax.set_xticklabels(x_tick_labels)

  # Style tick labels based on match frequency (green=match, red=mismatch, size scales with freq)
  for i, tick_label in enumerate(ax.get_xticklabels()):
    freq = match_freq[i] if i < len(match_freq) else 0
    # Color: interpolate from red (0) to green (1)
    tick_label.set_color((1 - freq, freq * 0.7, 0))  # RGB: red->green
    # Size: scale from 6 (no match) to 12 (full match)
    tick_label.set_fontsize(6 + freq * 6)
    # Weight: bold if high match
    tick_label.set_fontweight('bold' if freq > 0.5 else 'normal')
  plt.xlim(-.2,end_alignm-start_alignm-early_cut)

  for row in rows_of_pdbid:

    probing=row['reactivity']
    error=row['reactivity_errors']

    probing_slice = np.array(probing[start_alignm:end_alignm-early_cut], dtype=float)

    # Find NaN positions (marked as -1000)
    nan_mask = probing_slice == -1000
    nan_indices = np.where(nan_mask)[0]

    # Replace -1000 with 0 for plotting
    probing_clean = probing_slice.copy()
    probing_clean[nan_mask] = 0

    line_color = 'b' if row['experiment_type']=='2A3_MaP' else 'g'
    plt.plot(probing_clean, color=line_color, alpha=.6)

    # Mark NaN positions with X markers at y=0
    if len(nan_indices) > 0:
      plt.scatter(nan_indices, [0] * len(nan_indices), marker='x', s=50,
                  color=line_color, linewidths=2, zorder=5)
    # plt.scatter(range(end_alignm-start_alignm-early_cut),error[start_alignm:end_alignm-early_cut])


  # if early_cut:
    # plt.plot(pseudo_probing[:-early_cut])
  # else:
  plt.plot(pseudo_probing[0:end_alignm-start_alignm-early_cut],color='black')

  # plt.plot(range(end_alignm-start_alignm-early_cut),[-1.5]*(end_alignm-start_alignm-early_cut),color='gray', linestyle='dashed',)
  # plt.legend(['probing','errors','pseudo_probing'])
  plt.title(f'{pdb_id.upper()}')
  # plt.savefig(f"drive/MyDrive/probing-dataset/rnagym-rnapdb-probing-and-pseudoprobing-figs/{pdb_id}_{experiment_type}_{rnagym_id}.png")
  plt.show()

rnagym_seqs_aligned_with_pdb_id = rnagym_seqs[rnagym_seqs['pdb_id']=='3t4b'].reset_index(drop=True)
seen=[]
count=0
# probings_2a3=[]
# probings_dms=[]
rows_of_pdbid=[]
for idx, row in rnagym_seqs_aligned_with_pdb_id.iterrows():
  if (row['pdb_id'], row['sequence_id'], row['experiment_type']) in seen:
    continue
  seen.append((row['pdb_id'], row['sequence_id'], row['experiment_type']))
  count+=1
  # print(count)
  # print(row['sequence_id'])
  # print(row['reactivity'])
  # print(row['experiment_type'])
  # if row['experiment_type']=='2A3_MaP':
  #   probings_2a3.append(row['reactivity'])
  # elif row['experiment_type']=='DMS_MaP':
  #   probings_dms.append(row['reactivity'])
  rows_of_pdbid.append(row)

for row in rows_of_pdbid:
  # print(row['sequence_id'])
  compute_alignment_bounds(row['reactivity'], row['sequence'], row['seqB'])


alignm, start_alignm, end_alignm, draw = compute_alignment_bounds(rows_of_pdbid[0]['reactivity'], rows_of_pdbid[0]['sequence'], rows_of_pdbid[0]['seqB'])

plot_probings_and_pseudoprobing('3t4b', rows_of_pdbid, start_alignm, end_alignm, 0)