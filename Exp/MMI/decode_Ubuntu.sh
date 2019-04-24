#!/usr/bin/env bash

# MMI decode on Ubuntu

# Relative to project_root
DICT_FILE=/home/cgsdfc/JiweiLi_Ubuntu/ubuntu_20000

# Forward Model
MODEL_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/Ubuntu/MMI-forward/model
PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/Ubuntu/MMI-forward/params

# Backward Model
MMI_MODEL=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/Ubuntu/MMI-backward/model
MMI_PARAMS=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/Ubuntu/MMI-backward/params

INPUT_FILE=/home/cgsdfc/JiweiLi_Ubuntu/raw_testing_contexts.txt
SAVE_PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/Ubuntu/MMI-decoder/params
OUTPUT_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/Ubuntu/MMI-decoder/output.txt

BEAM_SIZE=200
MAX_LEN=20
BATCH_SIZE=20
GPU_INDEX=2

th Decode/decode.lua \
    -model_file ${MODEL_FILE} \
    -params_file ${PARAMS_FILE} \
    -gpu_index $GPU_INDEX \
    -OutputFile ${OUTPUT_FILE} \
    -dictPath ${DICT_FILE} \
    -save_params_file ${SAVE_PARAMS_FILE} \
    -MMI -MMI_model_file ${MMI_MODEL} \
    -MMI_params_file ${MMI_PARAMS} \
    -batch_size ${BATCH_SIZE} \
    -InputFile ${INPUT_FILE} \
    -beam_size $BEAM_SIZE \
    -max_length $MAX_LEN
