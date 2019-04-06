#!/usr/bin/env bash

# 3rd step: extract top most common response in the decoded responses.

ROUND=0
SAVE_ROOT=save/test-distill

OUTPUT_FILE=$SAVE_ROOT/$ROUND/tmp/top_response5.txt
INPUT_FILE=$SAVE_ROOT/$ROUND/tmp/decode.txt

python Distill/extract_top/select_top_decoded.py \
    $INPUT_FILE \
    $OUTPUT_FILE
