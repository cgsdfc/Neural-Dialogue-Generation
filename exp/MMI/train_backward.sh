#!/usr/bin/env bash

DATA_ROOT=/home/cgsdfc/OpenSubData/dialogue_length2_3
SAVE_ROOT=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation

BATCH_SIZE=20
DIMENSION=1000
DROP_OUT=0.2
INIT_WEIGHT=0.08
ALPHA=0.1
START_HALVE=5
LAYERS=4
GPU_INDEX=2
THRES=1
MAX_ITER=8

SAVE_FOLDER=$SAVE_ROOT/MMI-backward-dialen_2_3

TRAIN_FILE=$DATA_ROOT/s_given_t_dialogue_length2_3_train.txt
DEV_FILE=$DATA_ROOT/s_given_t_dialogue_length2_3_dev.txt
TEST_FILE=$DATA_ROOT/s_given_t_dialogue_length2_3_test.txt

th Atten/train.lua \
    -batch_size $BATCH_SIZE \
    -dimension $DIMENSION \
    -dropout $DROP_OUT \
    -init_weight $INIT_WEIGHT \
    -alpha $ALPHA \
    -layers $LAYERS \
    -gpu_index $GPU_INDEX \
    -thres $THRES \
    -saveFolder $SAVE_FOLDER \
    -train_file $TRAIN_FILE \
    -dev_file $DEV_FILE \
    -test_file $TEST_FILE \
    -max_iter $MAX_ITER \
    -start_halve $START_HALVE \
    -train_backward \
    -time_one_batch
