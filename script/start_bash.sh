#!/usr/bin/env bash

# Run bash in the torch container.

IMAGE=kaixhin/cuda-torch-mega:8.0

PROJECT_ROOT=/home/cgsdfc/deployment/Models/Neural-Dialogue-Generation
OPENSUB_DATA_ROOT=/home/cgsdfc/OpenSubData/
SAVE_ROOT=/home/cgsdfc/SavedModels/Neural-Dialogue-Generation/
UBUNTU_DATA_ROOT=/home/cgsdfc/JiweiLi_Ubuntu/


docker run --runtime nvidia --rm -it \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    -v $OPENSUB_DATA_ROOT:$OPENSUB_DATA_ROOT \
    -v $UBUNTU_DATA_ROOT:$UBUNTU_DATA_ROOT \
    -v $SAVE_ROOT:$SAVE_ROOT \
    -w ${PROJECT_ROOT} \
    ${IMAGE} \
    bash
