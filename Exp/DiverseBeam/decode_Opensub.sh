#!/usr/bin/env bash

# Relative to project_root
DICT_FILE=data/movie_25000

MODEL_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-forward-dialogue_length2_6/model
PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/MMI-forward-dialogue_length2_6/params
INPUT_FILE=/home/cgsdfc/JiweiLi_Ubuntu/raw_testing_contexts.txt
SAVE_PARAMS_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/DiverseBeam/Opensub/param
OUTPUT_FILE=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/DiverseBeam/Opensub/output.txt

BEAM_SIZE=10
MAX_LEN=20
BATCH_SIZE=20
DIVERSE_RATE=0.5 # Empirical value.

th Decode/decode.lua \
    -model_file ${MODEL_FILE} \
    -params_file ${PARAMS_FILE} \
    -gpu_index 1 \
    -OutputFile ${OUTPUT_FILE} \
    -dictPath ${DICT_FILE} \
    -save_params_file ${SAVE_PARAMS_FILE} \
    -batch_size ${BATCH_SIZE} \
    -InputFile ${INPUT_FILE} \
    -beam_size $BEAM_SIZE \
    -max_length $MAX_LEN \
    -DiverseRate $DIVERSE_RATE
