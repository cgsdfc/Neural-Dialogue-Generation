#!/usr/bin/env bash

# The second step in one data distillation iteration is decoding a large amount of example on
# the dataset using the seq2seq model trained in the last iteration.

ROUND=0
MAX_DECODED_NUM=1000000
BATCH_SIZE=128

SAVE_ROOT=save/test-distill/
MODEL_FILE=$SAVE_ROOT/$ROUND/model/model8
PARAMS_FILE=$SAVE_ROOT$ROUND/model/params

INPUT_FILE=$SAVE_ROOT/$ROUND/data/t_given_s_train.txt
OUTPUT_FILE=$SAVE_ROOT/$ROUND/tmp/decode.txt

test -f $MODEL_FILE || exit 1
test -f $INPUT_FILE || exit 1

th Decode/decode.lua \
        -params_file $PARAMS_FILE \
        -model_file $MODEL_FILE \
        -InputFile $INPUT_FILE \
        -OutputFile $OUTPUT_FILE \
        -batch_size ${BATCH_SIZE} \
        -max_decoded_num ${MAX_DECODED_NUM}
