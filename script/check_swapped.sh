#!/usr/bin/env bash

# This file verifies that the supposed-different file pairs are
# *actually* the same.
# Now the duplicates are cleanup, you should only run it to check the unmodified
# problematic data dir.

PREFIX=/home/cgsdfc/OpenSubData/

ITEMS=( 2_3 2_6 3_6 )

for item in ${ITEMS[@]}
do
    s_t=$PREFIX/s_given_t_dialogue_length$item.txt
    t_s=$PREFIX/t_given_s_dialogue_length$item.txt
    cmp $s_t $t_s
    echo "$?: $s_t $t_s"
done
