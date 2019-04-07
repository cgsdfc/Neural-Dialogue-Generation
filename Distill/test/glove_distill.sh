#!/usr/bin/env bash

# Distill based on already-computed glove scores.

ROUND=0
SAVE_ROOT=save/test-distill

SAVE_SCORE_FILE=$SAVE_ROOT/$ROUND/tmp/glove_score
TOP_RES_FILE=$SAVE_ROOT/$ROUND/tmp/top_response4.txt
TRAIN_DATA=$SAVE_ROOT/$ROUND/data/t_given_s_train.txt
SAVE_FOLDER=$SAVE_ROOT/$ROUND/glove/
BATCH_SIZE=1


th Distill/Glove/distill.lua \
    -TopResponseFile $TOP_RES_FILE \
    -TrainingData $TRAIN_DATA \
    -saveFolder $SAVE_FOLDER \
    -batch_size $BATCH_SIZE \
    -save_summary \
    -distill_four_gram
