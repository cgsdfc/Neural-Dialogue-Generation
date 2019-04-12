#!/usr/bin/env bash

# Run bash in the torch container.

IMAGE=kaixhin/cuda-torch-mega:8.0

PROJECT_ROOT=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation
DATA_ROOT=/home/cgsdfc/OpenSubData/
SAVE_ROOT=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/


docker run --runtime nvidia --rm -it \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    -v $DATA_ROOT:$DATA_ROOT \
    -v $SAVE_ROOT:$SAVE_ROOT \
    -w ${PROJECT_ROOT} \
    ${IMAGE} \
    bash
