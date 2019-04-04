#!/usr/bin/env bash

MODEL_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-forward/model8
PARAMS_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-forward/params
OUTPUT_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-decode/output.txt
DICT_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/data/movie_25000

th Decode/decode.lua -model_file ${MODEL_FILE} -params_file ${PARAMS_FILE} -gpu_index 2 \
    -OutputFile ${OUTPUT_FILE} -dictPath ${DICT_FILE}
