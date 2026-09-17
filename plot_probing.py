from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (15,3)
structure_and_probing=pd.read_csv(
  "https://raw.githubusercontent.com/lucianozablocki/probing-dataset/refs/heads/main/structure_and_probing.csv",
  converters={'reactivity': literal_eval}
)
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

def plot_probings_and_pseudoprobing(pdb_id, rows_of_pdbid, alignment_bounds, probings_to_plot=None, save_fig=False):
  """
  Plot probings and pseudo-probing for a given pdb_id.

  Args:
    pdb_id: The PDB ID to plot
    rows_of_pdbid: List of rows containing probing data
    alignment_bounds: List of (start_alignm, end_alignm) tuples, one per row in rows_of_pdbid
  """
  plt.figure()
  plt.grid()
  struct=rows_of_pdbid.iloc[0]['dot_bracket']
  ref_seq=rows_of_pdbid.iloc[0]['sequence']
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
  for local_idx, (global_idx, row) in enumerate(rows_of_pdbid.iterrows()):
    start_alignm, end_alignm = alignment_bounds[local_idx]
    seqA = row.aligned_rnagym_seq
    seqB_aligned = row.aligned_pdb_seq
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

  structures_idxs=0
  alpha = .6 if len(rows_of_pdbid) < 20 else .2
  for local_idx, (global_idx, row) in enumerate(rows_of_pdbid.iterrows()):
    print(f"Processing row {global_idx} for pdb_id {pdb_id}")
    if probings_to_plot is not None and global_idx not in probings_to_plot:
      continue
    start_alignm, end_alignm = alignment_bounds[local_idx]
    # print(f"align starts at {start_alignm} and ends at {end_alignm}")
    probing=row['reactivity']
    error=row['reactivity_errors']
    seqA_aligned = row.aligned_rnagym_seq
    seqB_aligned = row.aligned_pdb_seq

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

    line_color = 'b' if row['experiment']=='2A3_MaP' else 'g'
    plt.plot(x_positions, probing_clean, color=line_color, alpha=alpha)
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

  legend_handles = [
    Line2D([0], [0], color='g', label='DMS'),
    Line2D([0], [0], color='b', label='2A3'),
  ]
  plt.legend(handles=legend_handles)

  plt.title(f'{pdb_id.upper()} chain {chain}')
  if save_fig:
    plt.savefig(f"{pdb_id} chain {chain}.png")
  plt.show()
  print(f"{pdb_id}")
  return 0

grouped_df=structure_and_probing.groupby(['pdb_id','chain'])
for (pdb_id, chain), group in grouped_df:
  if pdb_id!='1xjr' or chain!='A':
    continue

  # Collect alignment bounds for each row
  alignment_bounds = []
  for idx,(_,row) in enumerate(group.iterrows()):
    start_alignm, end_alignm = find_alignment_bounds(row.aligned_pdb_seq)
    alignment_bounds.append((start_alignm, end_alignm))
  plot_probings_and_pseudoprobing(pdb_id, group, alignment_bounds, save_fig=True)