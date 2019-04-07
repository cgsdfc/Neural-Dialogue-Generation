#!/usr/bin/env bash

# Distill all three files: dev, train and test.

ROUND=0
SAVE_ROOT=save/test-distill

SAVE_SCORE_FILE=$SAVE_ROOT/$ROUND/tmp/glove_score
TOP_RES_FILE=$SAVE_ROOT/$ROUND/tmp/top_response4.txt
TRAIN_DATA_ROOT=$SAVE_ROOT/$ROUND/data/

TOTAL_LINES=256


TO_DISTILL=(
    t_given_s_dev.txt
    t_given_s_test.txt
    t_given_s_train.txt
)

OUTPUT_ROOT=$SAVE_ROOT/$ROUND/output

mkdir -p $OUTPUT_ROOT

for file in ${TO_DISTILL[@]}
do
    TRAIN_DATA=$TRAIN_DATA_ROOT/$file
    OUTPUT_FILE=$OUTPUT_ROOT/$file

    echo "Distilling $TRAIN_DATA"

    th Distill/Glove/distill.lua \
        -TopResponseFile $TOP_RES_FILE \
        -TrainingData $TRAIN_DATA \
        -OutputFile $OUTPUT_FILE \
        -total_lines $TOTAL_LINES \
        -load_score \
        -save_score_file $SAVE_SCORE_FILE
done

echo "all distillation done"
