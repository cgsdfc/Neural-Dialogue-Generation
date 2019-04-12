#!/usr/bin/env bash

# Test decoding with backward task.
# Requirements:
# 1. pretrained attention model, as Decode/ would.
# 2. pretrained soothsayer model predicting backward probability. (Backward/Model)
# 3. input file to decode.

SEQ2SEQ_PARAMS=save/test-atten/params
SEQ2SEQ_MODEL=save/test-atten/model8
TASK=backward

INPUT_FILE=data/t_given_s_test.txt
OUTPUT_FILE=save/test-future-$TASK-decode/decode.txt

SOOTHSAYER_MODEL=save/test-future-$TASK/model10
PREDICTOR_WEIGHT=1

th Future/Decode/decode.lua \
    -params_file $SEQ2SEQ_PARAMS \
    -model_file  $SEQ2SEQ_MODEL \
    -InputFile $INPUT_FILE \
    -OutputFile $OUTPUT_FILE \
    -FuturePredictorModelFile $SOOTHSAYER_MODEL \
    -PredictorWeight $PREDICTOR_WEIGHT \
    -Task $TASK \
    -gpu_index 2
