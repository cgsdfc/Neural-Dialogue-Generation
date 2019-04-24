#!/usr/bin/env bash

INPUT=(
    /home/cgsdfc/UbuntuDialogueCorpus/Test.dialogues.pkl
    /home/cgsdfc/UbuntuDialogueCorpus/Training.dialogues.pkl
    /home/cgsdfc/UbuntuDialogueCorpus/Validation.dialogues.pkl
)

PREFIX=/home/cgsdfc/JiweiLi_Ubuntu

for input in ${INPUT[@]} ; do
    python script/Ubuntu/convert.py --convert-data -p $PREFIX -i $input
done
