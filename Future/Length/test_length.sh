#!/usr/bin/env bash

PRAMS_FILE=save/test-atten/params
MODEL_FILE=save/test-atten/model8
SAVE_FOLDER=save/test-future-length/

th Future/Length/train.lua \
    -params_file $PRAMS_FILE \
    -model_file $MODEL_FILE \
    -gpu_index 2 \
    -save_model_path $SAVE_FOLDER
