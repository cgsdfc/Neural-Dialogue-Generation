#!/usr/bin/env bash

# Distill all three files: dev, train and test.

ROUND=0
SAVE_ROOT=save/test-distill

TOP_RES_FILE=$SAVE_ROOT/$ROUND/tmp/top_response5.txt
TRAIN_DATA_ROOT=$SAVE_ROOT/$ROUND/data/

# Glove or Encoder
DISTILL_METHOD=$1
BATCH_SIZE=1

TO_DISTILL=(
    t_given_s_dev.txt
    t_given_s_test.txt
    t_given_s_train.txt
)

OUTPUT_ROOT=$SAVE_ROOT/$ROUND/output

mkdir -p $OUTPUT_ROOT

echo "Using $DISTILL_METHOD"

for file in ${TO_DISTILL[@]}
do
    TRAIN_DATA=$TRAIN_DATA_ROOT/$file
    echo "Distilling $TRAIN_DATA"

    th Distill/$DISTILL_METHOD/distill.lua \
        -TopResponseFile $TOP_RES_FILE \
        -TrainingData $TRAIN_DATA \
        -saveFolder $OUTPUT_ROOT \
        -batch_size $BATCH_SIZE \
        -gpu_index 2 \
        -save_summary
done

echo "all distillation done"
