#!/usr/bin/env bash

BATCH_SIZE=128

DIMENSION=
DROP_OUT=0.2
INIT_WEIGHT=0.1
ALPHA=1.0
LAYERS=4
GPU_INDEX=1
SPEAKER_SETTING=
THRES=5

SAVE_FOLDER=
TRAIN_FILE=
DEV_FILE=
TEST_FILE=

th Persona/train.lua \
    -batch_size $BATCH_SIZE \
    -dimension $DIMENSION \
    -dropout $DROP_OUT \
    -init_weight $INIT_WEIGHT \
    -alpha $ALPHA \
    -layers $LAYERS \
    -gpu_index $GPU_INDEX \
    -speakerSetting $SPEAKER_SETTING \
    -thres $THRES \
    -saveFolder $SAVE_FOLDER \
    -train_file $TRAIN_FILE \
    -dev_file $DEV_FILE \
    -test_file $TEST_FILE



