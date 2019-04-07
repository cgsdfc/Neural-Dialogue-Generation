#!/usr/bin/env bash

# The 4th step: distill the data according to the similarity score.
ROUND=0
SAVE_ROOT=save/test-distill

TOP_RES_FILE=$SAVE_ROOT/$ROUND/tmp/top_response5.txt
TRAIN_DATA=$SAVE_ROOT/$ROUND/data/t_given_s_train.txt
SAVE_FOLDER=$SAVE_ROOT/$ROUND/encoder

BATCH_SIZE=1
PARAMS_FILE=save/test-atten/params
MODEL_FILE=save/test-atten/model8


th Distill/Encoder/distill.lua \
    -TopResponseFile $TOP_RES_FILE \
    -TrainingData $TRAIN_DATA \
    -saveFolder $SAVE_FOLDER \
    -params_file $PARAMS_FILE \
    -model_file $MODEL_FILE \
    -batch_size $BATCH_SIZE \
    -gpu_index 2 \
    -save_score \
    -save_removed

