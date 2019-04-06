#!/usr/bin/env bash

# Distill based on already-computed glove scores.

ROUND=0
SAVE_ROOT=save/test-distill

SAVE_SCORE_FILE=$SAVE_ROOT/$ROUND/tmp/glove_score
TOP_RES_FILE=$SAVE_ROOT/$ROUND/tmp/top_response4.txt
TRAIN_DATA=$SAVE_ROOT/$ROUND/data/t_given_s_train.txt
OUTPUT_FILE=$SAVE_ROOT/$ROUND/tmp/distill.txt
TOTAL_LINES=256


th Distill/Glove/distill.lua \
    -TopResponseFile $TOP_RES_FILE \
    -TrainingData $TRAIN_DATA \
    -OutputFile $OUTPUT_FILE \
    -total_lines $TOTAL_LINES \
    -load_score \
    -save_score_file $SAVE_SCORE_FILE
