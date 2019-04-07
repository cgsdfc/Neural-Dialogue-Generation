#!/usr/bin/env bash

MODEL_FILE=./save/test-atten/model8
PARAMS_FILE=./save/test-atten/params
OUTPUT_FILE=./save/test-decode/output.txt
DICT_FILE=./data/movie_25000
SAVE_PARAMS_FILE=./save/test-decode/params

th Decode/decode.lua \
    -model_file ${MODEL_FILE} \
    -params_file ${PARAMS_FILE} \
    -gpu_index 2 \
    -OutputFile ${OUTPUT_FILE} \
    -dictPath ${DICT_FILE} \
    -save_params_file $SAVE_PARAMS_FILE
