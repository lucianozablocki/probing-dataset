import pandas as pd
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from Bio import pairwise2
from multiprocessing import Pool, cpu_count
import os
import logging
import numpy as np

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

# Configure logging to write to a single file (filename can be set via FMS_LOG_FILE env var)
LOG_FILE = os.environ.get("FMS_LOG_FILE", "find_matching_seqs.log")
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode='a',
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global variable for worker processes to access PDB data
_pdb_data = None
_max_len_rnapdb = None

def init_worker(pdb_file, max_len):
    """Initialize worker process with PDB data."""
    global _pdb_data, _max_len_rnapdb
    _pdb_data = pd.read_csv(pdb_file)
    _max_len_rnapdb = max_len
    logger.info(f"Worker initialized with {len(_pdb_data)} PDB sequences")

def process_chunk(args):
    """Process a single chunk against all PDB sequences."""
    chunk_idx, chunk_data = args
    
    logger.info(f"########## Processing CHUNK NUMBER {chunk_idx} ##########")
    
    results = []
    
    total_seqs = len(chunk_data)
    seq_counter = 0  # Counter for sequences within this chunk
    
    for idx2, record2 in _pdb_data.iterrows():
        if idx2 % 500 == 0:
            logger.info(f"Chunk {chunk_idx}: processing PDB {idx2} of {len(_pdb_data)}")
        
        seqB = normalize_sequence(record2['sequence'])
        
        for _, record in chunk_data.iterrows():
            seq_counter += 1
            # Log progress periodically, every 10000 sequences
            if seq_counter % 10000 == 0:
                logger.info(
                    f"Chunk {chunk_idx}: processed {seq_counter} of {CHUNKSIZE} train sequences against PDB {record2['pdbid']}"
                )
            
            seqA = normalize_sequence(record['sequence'])
            
            alignm = pairwise2.align.localds(seqA, seqB, match_dic, GapOpen, GapExtend, one_alignment_only=True)[0]
            equal_nucleotides_count = sum(a == b for a, b in zip(alignm.seqA, alignm.seqB))
            max_len_local = max(len(seqB), len(seqA))
            local_IDscore_bymax = equal_nucleotides_count / max_len_local
            global_IDscore_bymax = equal_nucleotides_count / _max_len_rnapdb
            lens_array = [len(seqA), len(seqB)]
            min_pos = np.argmin(lens_array)
            min_len_local = lens_array[min_pos]
            local_IDscore_bymin = equal_nucleotides_count / min_len_local
            if (local_IDscore_bymin >=0.5) and ((min_len_local/lens_array[1-min_pos])>=0.2):
                logger.info(
                    f"found min len local score >= 0.5 for train seq id: {record['sequence_id']} and pdb id: {record2['pdbid']} at chunk {chunk_idx}"
                )
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
    
    logger.info(f"Chunk {chunk_idx}: completed with {len(results)} results")
    return results

def chunk_generator(filename, chunksize):
    """Generator that yields chunks one at a time to avoid loading all into memory."""
    with pd.read_csv(filename, chunksize=chunksize) as reader:
        for idx, chunk in enumerate(reader):
            yield (idx, chunk)

filename_1="./train_data.csv"
filename_2="./sanitized_rnapdbdataset.csv"
output_file="./train_vs_rnapdb_matches_alignment.csv"

max_len_rnapdb=470 # max_len_global
CHUNKSIZE = 10**3  # 1000 rows per chunk (adjust as needed)

# Determine number of workers (leave one CPU free for the main process)
num_workers = max(1, cpu_count() - 1)
logger.info(f"Using {num_workers} worker processes")
logger.info(f"Using chunksize of {CHUNKSIZE} rows per chunk")

first_write = True  # Set to True to create new file
chunks_processed = 0

# Process chunks in parallel using a generator (memory efficient)
logger.info("Starting parallel processing...")
logger.info(f"About to create Pool with {num_workers} workers...")

try:
    with Pool(processes=num_workers, initializer=init_worker, initargs=(filename_2, max_len_rnapdb)) as pool:
        logger.info("Pool created successfully, starting imap...")
        # imap processes chunks as they're generated, keeping only a few in memory at a time
        # chunksize=1 means submit one chunk at a time to workers
        chunk_iter = chunk_generator(filename_1, CHUNKSIZE)
        logger.info("Generator created, starting to process chunks...")
        
        for chunk_results in pool.imap(process_chunk, chunk_iter, chunksize=1):
            chunks_processed += 1
            logger.info(
                f"Main process: received {len(chunk_results)} results from chunk (total chunks processed: {chunks_processed})"
            )

            # Write results for this chunk directly to disk (no intermediate buffer)
            if chunk_results:
                logger.info(f"Writing {len(chunk_results)} results from chunk {chunks_processed} to {output_file}...")
                df_batch = pd.DataFrame(chunk_results)
                if first_write:
                    df_batch.to_csv(output_file, index=False, mode='w')
                    first_write = False
                else:
                    df_batch.to_csv(output_file, index=False, mode='a', header=False)
                logger.info("Chunk written successfully")
except Exception as e:
    logger.exception(f"ERROR in parallel processing: {e}")
    raise

logger.info(f"Finished writing all results to {output_file}. Total chunks processed: {chunks_processed}")