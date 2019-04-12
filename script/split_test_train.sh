#!/usr/bin/env bash


DATASETS=(
    /home/cgsdfc/OpenSubData/s_given_t_dialogue_length2_3.txt
    /home/cgsdfc/OpenSubData/s_given_t_dialogue_length2_6.txt
    /home/cgsdfc/OpenSubData/s_given_t_dialogue_length3_6.txt
)

PREFIXES=(
    /home/cgsdfc/OpenSubData/dialogue_length2_3
    /home/cgsdfc/OpenSubData/dialogue_length2_6
    /home/cgsdfc/OpenSubData/dialogue_length3_6
)

for (( i = 0; i < ${#DATASETS[@]}; ++i )); do
    dataset=${DATASETS[$i]}
    prefix=${PREFIXES[$i]}

    if test ! -d $prefix
    then
        mkdir $prefix
    fi

    # default ratio just fine.
    python script/data/split.py $dataset $prefix
done
