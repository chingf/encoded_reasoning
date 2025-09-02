#!/bin/bash

# Define batch size variable
batch_size=750

# Calculate max_samples (step size) as batch_size * 10
max_samples=$((batch_size * 10))

for idx in $(seq 54000 $max_samples 120000); do
    echo "Running with start_idx: $idx"
    python 02_generate_transcript_llama3-70b_deepseek_distill.py --start_idx $idx --batch_size $batch_size --max_samples $max_samples
    
    # Check if the previous command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed start_idx: $idx"
    else
        echo "Error occurred with start_idx: $idx"
        exit 1  # Stop on first error
    fi
    
    echo "---"
done