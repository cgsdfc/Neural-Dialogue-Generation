#!/usr/bin/env bash

# Test training of the attention model.
SAVE_FOLDER=save/test-atten

th Atten/train.lua \
    -gpu_index 2 \
    -saveFolder ${SAVE_FOLDER}
