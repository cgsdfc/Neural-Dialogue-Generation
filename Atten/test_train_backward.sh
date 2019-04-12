#!/usr/bin/env bash

# Test test_backward feature.
# This allows the source and target of an example to be effectively swapped
# *on-the-fly* when the dataset is loaded into memory.
# There is no need to prepare one forward and one backward dataset any more.

SAVE_FOLDER=save/test-atten-train-backward

th Atten/train.lua \
    -gpu_index 2 \
    -saveFolder ${SAVE_FOLDER} \
    -train_backward
