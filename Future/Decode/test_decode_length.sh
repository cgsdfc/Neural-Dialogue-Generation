#!/usr/bin/env bash

SEQ2SEQ_PARAMS=save/test-atten/params
SEQ2SEQ_MODEL=save/test-atten/model8

INPUT_FILE=data/t_given_s_test.txt
OUTPUT_FILE=save/test-future-length-decode/decode.txt

SOOTHSAYER_MODEL=save/test-future-length/model25
PREDICTOR_WEIGHT=1
TASK=length
TARGET_LENGTH=20

th Future/Decode/decode.lua \
    -params_file $SEQ2SEQ_PARAMS \
    -model_file  $SEQ2SEQ_MODEL \
    -InputFile $INPUT_FILE \
    -OutputFile $OUTPUT_FILE \
    -FuturePredictorModelFile $SOOTHSAYER_MODEL \
    -PredictorWeight $PREDICTOR_WEIGHT \
    -Task $TASK \
    -target_length $TARGET_LENGTH \
    -gpu_index 2
