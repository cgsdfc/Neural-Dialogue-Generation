#!/usr/bin/env bash

# This 1st step in the data distillation is to train a standard Seq2Seq model on
# the dataset resulted from the last round of distillation.
# The model is then used to distill what it has been trained on and produce the dataset
# for the next round.

ROUND=0
SAVE_ROOT=save/test-distill/

TRAIN_FILE=${SAVE_ROOT}/${ROUND}/data/t_given_s_train.txt
DEV_FILE=${SAVE_ROOT}/${ROUND}/data/t_given_s_dev.txt
TEST_FILE=${SAVE_ROOT}/${ROUND}/data/t_given_s_test.txt
SAVE_FOLDER=${SAVE_ROOT}/${ROUND}/model

test -f $TRAIN_FILE || exit 1

mkdir -p ${SAVE_ROOT}/${ROUND}/data
mkdir -p ${SAVE_ROOT}/${ROUND}/model

th Atten/train_atten.lua \
    -train_file ${TRAIN_FILE} \
    -dev_file ${DEV_FILE} \
    -test_file ${TEST_FILE} \
    -saveFolder ${SAVE_FOLDER}
