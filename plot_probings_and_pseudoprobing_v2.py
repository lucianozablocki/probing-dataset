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

# def compute_alignment_bounds(probing, seqA, seqB, print_seqB=False, print_seqA=False):
#   for idx, p in enumerate(probing):
#     # print(p)
#     # print(probing[idx+1])
#     if idx==len(probing)-1:
#       # print(f"probing values ends at {idx+1}")
#       break
#     if p==float(-1000) and probing[idx+1]!=float(-1000):
#       # print(f"probing values starts at {idx+1}")
#       probing_start=idx+1
#     if p!=float(-1000) and probing[idx+1]==float(-1000):
#       # print(f"probing values ends at {idx}")
#       probing_end=idx

#   alignm = pairwise2.align.localds(seqA, seqB, match_dic, GapOpen, GapExtend, one_alignment_only=True)[0]
#   lens_array = [len(seqA), len(seqB)]
#   min_pos = np.argmin(lens_array)
#   min_len_local = lens_array[min_pos]
#   equal_nucleotides_count = sum(a == b for a, b in zip(alignm.seqA, alignm.seqB))
#   local_IDscore_bymin = equal_nucleotides_count / min_len_local

#   # First pass: find start_alignm and end_alignm
#   not_start=True
#   start_alignm = None
#   end_alignm = None
#   for idx, (nucleotideA, nucleotideB) in enumerate(zip(alignm.seqA,alignm.seqB)):
#     if nucleotideB != "-":
#       if not_start:
#         start_alignm = idx
#         not_start = False
#       end_alignm = idx  # Always update end to the last non-gap position

#   if print_seqA:
#     # Second pass: print from start_alignm onwards
#     for idx in range(start_alignm, end_alignm + 1):
#       nucleotideA = alignm.seqA[idx]
#       nucleotideB = alignm.seqB[idx]
#       if nucleotideA == nucleotideB:
#         print(f"{BOLD}{nucleotideA}{RESET}",end="")
#       else:
#         print(f"{nucleotideA.lower()}",end="")
#     print("")
  
#   if print_seqB:
#     print(alignm.seqB[start_alignm:end_alignm + 1])

#   # print("-"*start_alignm,end="")

#   # for idx, nucleotideB in enumerate(seqB):
#   #   # if nucleotideB=='-':
#   #   #   print("-",end="")
#   #   # else:
#   #   print(f"{idx%10}",end="")

#   # print("-"*(len(alignm.seqA)-len(seqB)-start_alignm),end="")
#   # print("\n")

#   # print(f"alignment starts at {start_alignm}")
#   # print(f"alignment ends at {end_alignm}")


#   # if start_alignm < probing_start:
#   #   # draw=False
#   #   start_alignm=probing_start
#   #   draw=True
#   # elif end_alignm > probing_end:
#   #   end_alignm=probing_end
#   #   draw=True
#   # else:
#   #   draw=True

#   return alignm, start_alignm, end_alignm, True

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,3)

def plot_probings_and_pseudoprobing(pdb_id, rows_of_pdbid, alignment_bounds, save_fig=False):
  """
  Plot probings and pseudo-probing for a given pdb_id.

  Args:
    pdb_id: The PDB ID to plot
    rows_of_pdbid: List of rows containing probing data
    alignment_bounds: List of (start_alignm, end_alignm) tuples, one per row in rows_of_pdbid
  """
  plt.figure()
  plt.grid()
  # the aligned sequence is the same in all alignments file row, just take the first
  # ^ ABOVE IS NOT ALWAYS TRUE, THERE ARE MULTIPLE CHAINS FOR EACH PBD ID
  ref_seq = rows_of_pdbid[0]['seqB']
  # `rnapdb_rows` holds all RNA PDB dataset rows for the given pdb_id,
  # we need to find the one that matches the aligned sequence to get the correct structure
  rnapdb_rows=rnapdb_dataset[rnapdb_dataset['id']==pdb_id]
  struct=None
  # before, we were taking the first sequence from all the possible chains.
  # changed to iterate over all pdb rows
  for idx, rnapdb_row in rnapdb_rows.iterrows():
    # basically find which of all the chains the alignment was run against
    if rnapdb_row['sequence']==ref_seq:
      struct=rnapdb_row['base_pairs']
  # print(f"{pdb_id}")
  # print(f"{struct}")
  if not struct:
    # at this point, we now that the reference sequence from the alignment doesn't match any of the sequences in rnapdb for this pdb_id,
    # so we can't get the correct structure to plot the pseudo-probing
    # and mainly because we used different tools to get the structure (rnapdbee) vs to get the sequences (rnaglib)
    print(f"reference seq differs between rnaglib and rnapdbee {pdb_id},")
    # anyways, they differ in at most a few nucleotides, so we can either
    # 1) run the alignment again for the "new sequence"
    # 2) put some kind of marker in the "not matching nucleotides", so in the plot we see that they differ
    return 2
  # print(struct)
  pseudo_probing=[1 if s=="." else 0 for s in list(struct)]
  pseudo_probing=np.array(pseudo_probing)-1.5

  # Use seqB (reference PDB sequence) length as the global range
  # print(ref_seq)
  # print(rnapdb_dataset[rnapdb_dataset['id']==pdb_id]['sequence'].values[0])
  seq_len = len(ref_seq)
  if len(struct)!=len(ref_seq):
    print(f"reference struct differs in length for pdb {pdb_id}")
    return False
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

  has_seqB_gap = False
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

    # Check if this row's alignment has seqB gaps
    if seq_len != len(probing_mapped) + len(gap_positions):
      has_seqB_gap = True
      break

    probing_slice = np.array(probing_mapped, dtype=float)

    # Find NaN positions (marked as -1000)
    nan_mask = probing_slice == -1000
    nan_indices = np.array(x_positions)[nan_mask]

    # Replace -1000 with 0 for plotting
    probing_clean = probing_slice.copy()
    probing_clean[nan_mask] = 0

    line_color = 'b' if row['experiment_type']=='2A3_MaP' else 'g'
    plt.plot(x_positions, probing_clean, color=line_color, alpha=.6 if pdb_id_counts[pdb_id]<20 else .2)
    # pearson_coef, p_value = pearsonr(probing_clean[x_positions], np.array([1 if s=="." else 0 for s in list(struct)]))
    # print(f"pearson coef: {pearson_coef}")
    # print(f"p_value: {p_value}")
    # Mark NaN positions and gap positions with X markers at y=0
    all_gap_indices = list(nan_indices) + gap_positions
    if len(all_gap_indices) > 0:
      plt.scatter(all_gap_indices, [0] * len(all_gap_indices), marker='x', s=50,
                  color=line_color, linewidths=2, zorder=5)
  if has_seqB_gap:
    print(f"there's probably gaps in seqB for pdb id {pdb_id}, skipping plot")
    return 1
  plt.plot(pseudo_probing[0:seq_len],color='black')
  plt.title(f'{pdb_id.upper()}')
  if save_fig:
    plt.savefig(f"/content/drive/MyDrive/probing-dataset/rnagym-rnapdb-overlapped-probing-figs-gapseqB/{pdb_id}.png")
  plt.show()
  return 0
