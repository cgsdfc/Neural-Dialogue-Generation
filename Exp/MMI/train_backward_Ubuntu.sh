#!/usr/bin/env bash


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
MAX_LEN=51
VOCAB_SIZE=20010

SAVE_FOLDER=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/Ubuntu/MMI-backward
TRAIN_FILE=/home/cgsdfc/JiweiLi_Ubuntu/Training.dialogues.txt
DEV_FILE=/home/cgsdfc/JiweiLi_Ubuntu/Validation.dialogues.txt
TEST_FILE=/home/cgsdfc/JiweiLi_Ubuntu/Test.dialogues.txt
DICK_FILE=/home/cgsdfc/JiweiLi_Ubuntu/ubuntu_20000


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
    -restart \
    -source_max_length $MAX_LEN \
    -target_max_length $MAX_LEN \
    -dictPath $DICK_FILE \
    -vocab_source $VOCAB_SIZE \
    -vocab_target $VOCAB_SIZE

