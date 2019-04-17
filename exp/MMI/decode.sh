#!/usr/bin/env bash

# Relative to project_root
DICT_FILE=data/movie_25000

# Forward Model
MODEL_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-forward-dialogue_length2_6/model
PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-forward-dialogue_length2_6/params

# Backward Model
MMI_MODEL=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-backward-dialogue_length2_6/model
MMI_PARAMS=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-backward-dialogue_length2_6/params

SAVE_PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-decoder/params
OUTPUT_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-decoder/output.txt

BATCH_SIZE=20

th Decode/decode.lua \
    -model_file ${MODEL_FILE} \
    -params_file ${PARAMS_FILE} \
    -gpu_index 2 \
    -OutputFile ${OUTPUT_FILE} \
    -dictPath ${DICT_FILE} \
    -save_params_file ${SAVE_PARAMS_FILE} \
    -MMI -MMI_model_file ${MMI_MODEL} \
    -MMI_params_file ${MMI_PARAMS} \
    -batch_size ${BATCH_SIZE}
