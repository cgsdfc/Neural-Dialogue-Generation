#!/usr/bin/env bash

# Run bash in the torch container.

IMAGE=kaixhin/cuda-torch-mega:8.0
PROJECT_ROOT=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation
OPEN_SUB_DATA_DIR=

docker run --runtime nvidia --rm -it \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    -w ${PROJECT_ROOT} \
    ${IMAGE} \
    bash
