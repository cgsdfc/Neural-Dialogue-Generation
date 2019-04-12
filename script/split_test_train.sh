#!/usr/bin/env bash


DATASETS=(
    /home/cgsdfc/OpenSubData/s_given_t_dialogue_length2_3.txt
    /home/cgsdfc/OpenSubData/s_given_t_dialogue_length2_6.txt
    /home/cgsdfc/OpenSubData/s_given_t_dialogue_length3_6.txt
    /home/cgsdfc/OpenSubData/t_given_s_dialogue_length2_3.txt
    /home/cgsdfc/OpenSubData/t_given_s_dialogue_length2_6.txt
    /home/cgsdfc/OpenSubData/t_given_s_dialogue_length3_6.txt
)

PREFIXES=(
    /home/cgsdfc/OpenSubData/2_3
    /home/cgsdfc/OpenSubData/2_6
    /home/cgsdfc/OpenSubData/3_6
    /home/cgsdfc/OpenSubData/2_3
    /home/cgsdfc/OpenSubData/2_6
    /home/cgsdfc/OpenSubData/3_6
)

for (( i = 0; i < ${#DATASETS[@]}; ++i )); do
    dataset=${DATASETS[$i]}
    prefix=${PREFIXES[$i]}
    python script/data/split.py $dataset $prefix
done
