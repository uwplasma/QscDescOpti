#!/bin/bash

# Define the name of the output file
output_file="total_combined.csv"

# Initialize output file
# Extract the header from the first file and write it to the output file
first_file=true

# Loop through all sub_output_*.csv files
for file in sub_output_*.csv; do
    if [ "$first_file" = true ]; then
        # Copy the entire content of the first file, including the header, to the output file
        cat "$file" > "$output_file"
        first_file=false
    else
        # From the second file onward, append only the data (skip the header)
        tail -n +2 "$file" >> "$output_file"
    fi
done

echo "All files have been merged into $output_file"
