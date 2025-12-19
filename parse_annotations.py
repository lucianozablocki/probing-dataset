#!/usr/bin/env python3
"""
Parse annotation files from the annotations folder and create a CSV with:
id, chain, sequence, base_pairs
"""

import os
import csv
from pathlib import Path


def parse_annotation_file(filepath):
    """Parse a single annotation file and yield (chain, sequence, base_pairs) tuples."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for strand header lines like >strand_QA
        if line.startswith('>strand_'):
            chain = line.replace('>strand_', '')
            
            # Next line is the sequence
            if i + 1 < len(lines):
                sequence = lines[i + 1].strip()
            else:
                sequence = ''
            
            # Line after that is the structure (base pairs)
            if i + 2 < len(lines):
                base_pairs = lines[i + 2].strip()
            else:
                base_pairs = ''
            
            yield (chain, sequence, base_pairs)
            i += 3
        else:
            i += 1


def main():
    annotations_dir = Path('annotations')
    output_file = 'rna_pdb_dataset_bp.csv'
    
    rows = []
    
    # Process all .txt files in annotations folder
    for txt_file in sorted(annotations_dir.glob('*.txt')):
        pdb_id = txt_file.stem  # filename without extension
        
        for chain, sequence, base_pairs in parse_annotation_file(txt_file):
            rows.append({
                'id': pdb_id,
                'chain': chain,
                'sequence': sequence,
                'base_pairs': base_pairs
            })
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'chain', 'sequence', 'base_pairs'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Processed {len(set(r['id'] for r in rows))} files")
    print(f"Total {len(rows)} chains written to {output_file}")


if __name__ == '__main__':
    main()
