#!/usr/bin/env bash

# Compute glove score as a precondition of distillation using glove.

BATCH_SIZE=64

ROUND=0
SAVE_ROOT=save/test-distill

SAVE_SCORE_FILE=$SAVE_ROOT/$ROUND/tmp/glove_score
TOP_RES_FILE=$SAVE_ROOT/$ROUND/tmp/top_response4.txt
TRAIN_DATA=$SAVE_ROOT/$ROUND/data/t_given_s_train.txt
OUTPUT_FILE=$SAVE_ROOT/$ROUND/tmp/distill.txt


th Distill/Glove/distill.lua \
    -TopResponseFile $TOP_RES_FILE \
    -TrainingData  $TRAIN_DATA \
    -OutputFile  $OUTPUT_FILE \
    -save_score \
    -save_score_file $SAVE_SCORE_FILE \
    -batch_size $BATCH_SIZE \
    -gpu_index 2
