#!/usr/bin/env bash

# Test training of the attention model.
SAVE_FOLDER=save/test-atten
BS=10

th Atten/train.lua \
    -gpu_index 1 \
    -saveFolder ${SAVE_FOLDER} \
    -batch_size $BS
