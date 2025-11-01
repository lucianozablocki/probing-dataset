import pandas as pd
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from Bio import pairwise2
from multiprocessing import Pool, cpu_count
import os

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

# Global variable for worker processes to access PDB data
_pdb_data = None
_max_len_rnapdb = None

def init_worker(pdb_file, max_len):
    """Initialize worker process with PDB data."""
    global _pdb_data, _max_len_rnapdb
    _pdb_data = pd.read_csv(pdb_file)
    _max_len_rnapdb = max_len
    print(f"Worker initialized with {len(_pdb_data)} PDB sequences", flush=True)

def process_chunk(args):
    """Process a single chunk against all PDB sequences."""
    chunk_idx, chunk_data = args
    
    print(f"########## Processing CHUNK NUMBER {chunk_idx} ##########", flush=True)
    
    results = []
    
    total_seqs = len(chunk_data)
    seq_counter = 0  # Counter for sequences within this chunk
    
    for idx2, record2 in _pdb_data.iterrows():
        if idx2 % 10 == 0:
            print(f"Chunk {chunk_idx}: processing PDB {idx2} of {len(_pdb_data)}", flush=True)
        
        seqB = normalize_sequence(record2['sequence'])
        
        if '&' in seqB:
            print(f"Chunk {chunk_idx}: skipping pdb seq with id: {record2['pdbid']} due to invalid character &", flush=True)
            continue
        if len(seqB) > 500 or len(seqB) < 10:
            print(f"Chunk {chunk_idx}: skipping pdb seq with id: {record2['pdbid']} due to length {len(seqB)}", flush=True)
            continue
        
        for _, record in chunk_data.iterrows():
            seq_counter += 1
            # Log progress periodically, every 10000 sequences
            if seq_counter % 10000 == 0:
                print(f"Chunk {chunk_idx}: processed {seq_counter} of {total_seqs} train sequences against PDB {record2['pdbid']}", flush=True)
            
            seqA = normalize_sequence(record['sequence'])
            
            alignm = pairwise2.align.localds(seqA, seqB, match_dic, GapOpen, GapExtend, one_alignment_only=True)[0]
            equal_nucleotides_count = sum(a == b for a, b in zip(alignm.seqA, alignm.seqB))
            max_len_local = max(len(seqB), len(seqA))
            #ver distribucion del max len local
            local_IDscore_bymax = equal_nucleotides_count / max_len_local
            global_IDscore_bymax = equal_nucleotides_count / _max_len_rnapdb
            min_len_local = min(len(seqB), len(seqA))
            local_IDscore_bymin = equal_nucleotides_count / min_len_local
            if min_len_local >0.5:
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
                    'alignment_seqA': alignm.seqA,
                    'alignment_seqB': alignm.seqB
                })
            # print(f"chunk {chunk_idx}: finished analyzing train seq id: {record['sequence_id']}", flush=True)
    
    print(f"Chunk {chunk_idx}: completed with {len(results)} results", flush=True)
    return results

def chunk_generator(filename, chunksize):
    """Generator that yields chunks one at a time to avoid loading all into memory."""
    with pd.read_csv(filename, chunksize=chunksize) as reader:
        for idx, chunk in enumerate(reader):
            yield (idx, chunk)

filename_1="./train_data.csv"
filename_2="./rna_pdb_dataset.csv"
output_file="./train_vs_rnapdb_matches_alignment.csv"

max_len_rnapdb=470 # max_len_global
CHUNKSIZE = 10**3  # 1,000 rows per chunk (adjust as needed)

# Determine number of workers (leave one CPU free for the main process)
num_workers = max(1, cpu_count() - 1)
print(f"Using {num_workers} worker processes", flush=True)
print(f"Using chunksize of {CHUNKSIZE} rows per chunk", flush=True)

write_batch_size = 100  # Write results every 100 alignments
first_write = True  # Set to True to create new file
results_buffer = []
chunks_processed = 0

# Process chunks in parallel using a generator (memory efficient)
print("Starting parallel processing...", flush=True)
print(f"About to create Pool with {num_workers} workers...", flush=True)

try:
    with Pool(processes=num_workers, initializer=init_worker, initargs=(filename_2, max_len_rnapdb)) as pool:
        print("Pool created successfully, starting imap...", flush=True)
        # imap processes chunks as they're generated, keeping only a few in memory at a time
        # chunksize=1 means submit one chunk at a time to workers
        chunk_iter = chunk_generator(filename_1, CHUNKSIZE)
        print("Generator created, starting to process chunks...", flush=True)
        
        for chunk_results in pool.imap(process_chunk, chunk_iter, chunksize=1):
            chunks_processed += 1
            print(f"Main process: received {len(chunk_results)} results from chunk (total chunks processed: {chunks_processed})", flush=True)
            
            # Add results from this chunk to buffer
            results_buffer.extend(chunk_results)
            
            # Write results incrementally
            if len(results_buffer) >= write_batch_size:
                print(f"Writing batch of {len(results_buffer)} results to {output_file}...", flush=True)
                df_batch = pd.DataFrame(results_buffer)
                if first_write:
                    df_batch.to_csv(output_file, index=False, mode='w')
                    first_write = False
                else:
                    df_batch.to_csv(output_file, index=False, mode='a', header=False)
                results_buffer = []  # Clear the buffer
                print(f"Batch written successfully", flush=True)
except Exception as e:
    print(f"ERROR in parallel processing: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise

# Write any remaining results
if results_buffer:
    print(f"Writing final batch of {len(results_buffer)} results to {output_file}...", flush=True)
    df_batch = pd.DataFrame(results_buffer)
    if first_write:
        df_batch.to_csv(output_file, index=False, mode='w')
    else:
        df_batch.to_csv(output_file, index=False, mode='a', header=False)

print(f"Finished writing all results to {output_file}. Total chunks processed: {chunks_processed}")