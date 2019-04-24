#!/usr/bin/env bash

PREFIX=/home/cgsdfc/JiweiLi_Ubuntu
INPUT=(
    /home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt
    /home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt
)

for input in ${INPUT[@]}; do
    python script/Ubuntu/truncate.py -p $PREFIX -i $input
done
