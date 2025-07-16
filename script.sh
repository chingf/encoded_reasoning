#!/bin/bash

for idx in {0..120000..5000}; do
    echo "Running with start_idx: $idx"
    python 02_generate_transcript_llama3-70b_deepseek_distill.py --start_idx $idx
    
    # Check if the previous command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed start_idx: $idx"
    else
        echo "Error occurred with start_idx: $idx"
        exit 1  # Stop on first error
    fi
    
    echo "---"
done