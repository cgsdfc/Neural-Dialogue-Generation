#!/usr/bin/env bash

# Test training of the backward attention model.
# Being *backward* means predicting the *source* given the *target*.
# the dataset files should have prefix *s_given_t*.
# The backward model is needed in MMI decoding.

SAVE_FOLDER=./save/test-atten-backward

th Atten/train_atten.lua -gpu_index 2 \
    -train_file ./data/s_given_t_train.txt \
    -dev_file ./data/s_given_t_train.txt \
    -test_file ./data/s_given_t_train.txt \
    -saveFolder ${SAVE_FOLDER}
