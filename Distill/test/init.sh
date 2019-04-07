#!/usr/bin/env bash

# Prepare the initial data

SAVE_ROOT=save/test-distill
ROUND=0
TARGET_ROOT=$SAVE_ROOT/$ROUND/data/
SOURCE_ROOT=data/

DATASET_FILES=(
    t_given_s_dev.txt
    t_given_s_test.txt
    t_given_s_train.txt
)

mkdir -p $TARGET_ROOT

# copy dataset files
for file in ${DATASET_FILES[@]}
do
    cp $SOURCE_ROOT/$file $TARGET_ROOT
done
