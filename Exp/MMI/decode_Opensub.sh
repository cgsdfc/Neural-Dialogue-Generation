#!/usr/bin/env bash

# MMI decode on OpenSubtitles.
# Relative to project_root
DICT_FILE=data/movie_25000

# Forward Model
MODEL_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-forward-dialogue_length2_6/model
PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-forward-dialogue_length2_6/params

# Backward Model
MMI_MODEL=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-backward-dialogue_length2_6/model
MMI_PARAMS=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-backward-dialogue_length2_6/params

INPUT_FILE=/home/cgsdfc/OpenSubData/dialogue_length2_6/test.txt
SAVE_PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-decoder/params
OUTPUT_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-decoder/output.txt

BEAM_SIZE=200
MAX_LEN=20
BATCH_SIZE=20
GPU_INDEX=1

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
