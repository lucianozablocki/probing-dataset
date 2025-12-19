#!/bin/bash

# Directory containing the CIF files
INPUT_DIR="pdb_cif_gz_files"
# Output directory for annotation results
OUTPUT_DIR="annotations"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .cif.gz files
for cif_file in "$INPUT_DIR"/*.cif.gz; do
    # Extract PDB ID from filename (e.g., 1a9n.cif.gz -> 1a9n)
    filename=$(basename "$cif_file")
    pdb_id="${filename%.cif.gz}"
    
    echo "Processing: $pdb_id"
    
    # Run annotator and save output to txt file
    /home/lzablocki/miniconda3/bin/annotator "$cif_file" > "$OUTPUT_DIR/${pdb_id}.txt" 2>&1
done

echo "Done! Results saved in $OUTPUT_DIR/"
