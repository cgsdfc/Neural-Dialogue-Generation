#!/usr/bin/env bash

# Test training of the forward attention model.
# Being *forward* means predicting the *target* given the *source*.
# the dataset files should have prefix *t_given_s*.
# The forward model is also called a valina seq2seq model.

SAVE_FOLDER=./save/test-atten-forward

th Atten/train_atten.lua -gpu_index 2 \
    -train_file ./data/t_given_s_train.txt \
    -dev_file ./data/t_given_s_dev.txt \
    -test_file ./data/t_given_s_test.txt \
    -saveFolder ${SAVE_FOLDER}
