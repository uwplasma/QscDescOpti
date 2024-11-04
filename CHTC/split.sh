#!/bin/bash

input_file="pass_desc.csv"
lines_per_file=50
header=$(head -n 1 $input_file) 

total_lines=$(wc -l < $input_file)
total_lines=$((total_lines - 1))

count=1

> filenames.txt

tail -n +2 $input_file | split -l $lines_per_file - "temp_part_"

for file in temp_part_*; do
    new_filename="sub_dataset_${count}.csv"
    echo "$header" > temp_file
    cat $file >> temp_file
    mv temp_file "$new_filename"
    rm $file
    echo "$new_filename" >> filenames.txt
    count=$((count + 1))
done

echo "Finish spliting, we have  $(( (total_lines + lines_per_file - 1) / lines_per_file )) subfiles now"
