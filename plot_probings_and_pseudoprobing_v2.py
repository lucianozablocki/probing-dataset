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

def plot_probings_and_pseudoprobing(pdb_id, rows_of_pdbid, alignment_bounds):
  """
  Plot probings and pseudo-probing for a given pdb_id.
  
  Args:
    pdb_id: The PDB ID to plot
    rows_of_pdbid: List of rows containing probing data
    alignment_bounds: List of (start_alignm, end_alignm) tuples, one per row in rows_of_pdbid
  """
  plt.figure()
  plt.grid()
  struct=rnapdb_dataset[rnapdb_dataset['id']==pdb_id]['base_pairs'].values[0]
  pseudo_probing=[1 if s=="." else 0 for s in list(struct)]
  pseudo_probing=np.array(pseudo_probing)-1.5

  # Use seqB (reference PDB sequence) length as the global range
  ref_seq = rows_of_pdbid[0]['seqB']
  seq_len = len(ref_seq)
  if len(struct)!=len(ref_seq):
    print(f"reference struct differs in length for pdb {pdb_id}")
    return
  # Collect all seqA alignments and compute match frequency per position
  match_counts = [0] * seq_len
  total_seqs = len(rows_of_pdbid)
  for idx, row in enumerate(rows_of_pdbid):
    start_alignm, end_alignm = alignment_bounds[idx]
    seqA = row.get('alignment_seqA')
    seqB_aligned = row.get('alignment_seqB')
    # Map aligned positions to seqB positions
    seqB_pos = 0
    for align_idx in range(start_alignm, min(end_alignm + 1, len(seqA))):
      if align_idx < len(seqB_aligned) and seqB_aligned[align_idx] != '-':
        if seqB_pos < seq_len and seqA[align_idx] == seqB_aligned[align_idx] and seqA[align_idx] != '-':
          match_counts[seqB_pos] += 1
        seqB_pos += 1

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
  plt.xlim(-.2, seq_len)

  for idx, row in enumerate(rows_of_pdbid):
    start_alignm, end_alignm = alignment_bounds[idx]
    # print(f"align starts at {start_alignm} and ends at {end_alignm}")
    probing=row['reactivity']
    error=row['reactivity_errors']
    seqA_aligned = row.get('alignment_seqA')
    seqB_aligned = row.get('alignment_seqB')

    # Map probing values to seqB positions
    # start_alignm is where seqB starts in the alignment, so it maps to seqB position 0
    # We need to track seqA position separately to handle gaps in seqA
    # Assumption: only seqA can have gaps
    probing_mapped = []
    x_positions = []
    gap_positions = []  # positions where seqA has a gap
    seqB_pos = 0
    seqA_pos = start_alignm

    for align_idx in range(start_alignm, min(end_alignm + 1, len(seqB_aligned))):
      if seqA_aligned[align_idx] == '-':
        # print(f"gap found at idx {align_idx}")
        # seqA has a gap - mark this position
        gap_positions.append(seqB_pos)
      else:
        # seqA has a nucleotide - use probing value
        probing_mapped.append(probing[seqA_pos])
        x_positions.append(seqB_pos) # positions where we have probing values (no gaps)
        seqA_pos += 1
      seqB_pos += 1

    probing_slice = np.array(probing_mapped, dtype=float)

    # Find NaN positions (marked as -1000)
    nan_mask = probing_slice == -1000
    nan_indices = np.array(x_positions)[nan_mask]

    # Replace -1000 with 0 for plotting
    probing_clean = probing_slice.copy()
    probing_clean[nan_mask] = 0

    line_color = 'b' if row['experiment_type']=='2A3_MaP' else 'g'
    plt.plot(x_positions, probing_clean, color=line_color, alpha=.6 if pdb_id_counts[pdb_id]<20 else .2)

    # Mark NaN positions and gap positions with X markers at y=0
    all_gap_indices = list(nan_indices) + gap_positions
    if len(all_gap_indices) > 0:
      plt.scatter(all_gap_indices, [0] * len(all_gap_indices), marker='x', s=50,
                  color=line_color, linewidths=2, zorder=5)

  plt.plot(pseudo_probing[0:seq_len],color='black')

  plt.title(f'{pdb_id.upper()}')
  if not (seq_len==len(probing_clean)+len(gap_positions)):
    print(f"there's probably gaps in seqB for pdb id {pdb_id}, skipping plot")
  else:
    plt.savefig(f"/content/drive/MyDrive/probing-dataset/rnagy-rnapdb-overlapped-probing-figs/{pdb_id}.png")
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

# Collect alignment bounds for each row
alignment_bounds = []
for row in rows_of_pdbid:
  alignm, start_alignm, end_alignm, draw = compute_alignment_bounds(row['reactivity'], row['sequence'], row['seqB'])
  alignment_bounds.append((start_alignm, end_alignm))

plot_probings_and_pseudoprobing('3t4b', rows_of_pdbid, alignment_bounds)