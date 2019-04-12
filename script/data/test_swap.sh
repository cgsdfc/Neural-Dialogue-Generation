#!/usr/bin/env bash

# Swapped twice should have no effect

INPUT=data/s_given_t_dev.txt
OUTPUT=/tmp/t_given_s_dev.txt
OUTPUT_2=/tmp/s_given_t_dev.txt

python script/data/swap.py $INPUT $OUTPUT
python script/data/swap.py $OUTPUT $OUTPUT_2
cmp $OUTPUT_2 $INPUT
echo $?
