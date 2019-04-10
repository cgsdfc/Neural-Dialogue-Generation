#!/usr/bin/env bash

SAVE_FOLDER=save/test-future-backward
FORWARD_PARAMS_FILE=save/test-atten-forward/params
FORWARD_MODEL_FILE=save/test-atten-forward/model8

BACKWARD_PARAMS_FILE=save/test-atten-backward/params
BACKWARD_MODEL_FILE=save/test-atten-backward/model8

th Future/Backward/train.lua \
    -save_model_path $SAVE_FOLDER \
    -forward_params_file $FORWARD_PARAMS_FILE \
    -forward_model_file $FORWARD_MODEL_FILE \
    -backward_params_file $BACKWARD_PARAMS_FILE \
    -backward_model_file $BACKWARD_MODEL_FILE \
    -gpu_index 2 \
    -batch_size 20
