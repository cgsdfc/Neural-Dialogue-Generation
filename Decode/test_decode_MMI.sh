#!/usr/bin/env bash

MODEL_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-forward/model8
PARAMS_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-forward/params
OUTPUT_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-decode-MMI/output.txt
DICT_FILE=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/data/movie_25000

MMI_MODEL=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-backward/model8
MMI_PARAMS=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-backward/params

th Decode/decode.lua -model_file ${MODEL_FILE} -params_file ${PARAMS_FILE} -gpu_index 2 \
    -OutputFile ${OUTPUT_FILE} -dictPath ${DICT_FILE} \
    -MMI -MMI_model_file ${MMI_MODEL} -MMI_params_file ${MMI_PARAMS}
