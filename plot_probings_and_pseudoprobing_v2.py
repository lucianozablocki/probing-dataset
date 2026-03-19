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
  # before, we were taking the first sequence from all the possible chains. changed to iterate over all pdb rows
  ref_seq = rows_of_pdbid[0]['seqB']
  # `rnapdb_rows` holds all rows for the given pdb_id,
  # we need to find the one that matches the aligned sequence to get the correct structure
  rnapdb_rows=rnapdb_dataset[rnapdb_dataset['id']==pdb_id]
  struct=None
  # print(rnapdb_rows)
  for idx, rnapdb_row in rnapdb_rows.iterrows():
    # print(rnapdb_row)
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
    return False
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
    # pearson_coef, p_value = pearsonr(probing_clean[x_positions], np.array([1 if s=="." else 0 for s in list(struct)]))
    # print(f"pearson coef: {pearson_coef}")
    # print(f"p_value: {p_value}")
    # Mark NaN positions and gap positions with X markers at y=0
    all_gap_indices = list(nan_indices) + gap_positions
    if len(all_gap_indices) > 0:
      plt.scatter(all_gap_indices, [0] * len(all_gap_indices), marker='x', s=50,
                  color=line_color, linewidths=2, zorder=5)
  plt.plot(pseudo_probing[0:seq_len],color='black')
  # assert(seq_len==len(probing_clean)+len(gap_positions))
  plt.title(f'{pdb_id.upper()}')
  if not (seq_len==len(probing_clean)+len(gap_positions)):
    print(f"there's probably gaps in seqB for pdb id {pdb_id}, skipping plot")
    return False
  else:
    if save_fig:
      plt.savefig(f"/content/drive/MyDrive/probing-dataset/rnagym-rnapdb-overlapped-probing-figs-gapseqB/{pdb_id}.png")
    plt.show()
    return True