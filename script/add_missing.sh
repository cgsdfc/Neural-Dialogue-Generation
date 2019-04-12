#!/usr/bin/env bash
#
INPUT=/home/cgsdfc/OpenSubData/t_given_s_dialogue_length3_6.txt
OUTPUT=/home/cgsdfc/OpenSubData/s_given_t_dialogue_length3_6.txt

python script/data/swap.py $INPUT $OUTPUT
