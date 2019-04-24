#!/usr/bin/env bash

DICT_PKL=/home/cgsdfc/UbuntuDialogueCorpus/Dataset.dict.pkl
OUTPUT=/home/cgsdfc/JiweiLi_Ubuntu/Dataset.dict.txt

python -m pickle $DICT_PKL >$OUTPUT

