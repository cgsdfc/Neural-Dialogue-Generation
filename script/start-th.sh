#!/usr/bin/env bash

# Run torch-REPL in the torch container.

IMAGE=kaixhin/cuda-torch-mega:8.0
PROJECT_ROOT=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation

docker run --runtime nvidia --rm -it \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    ${IMAGE} \
    th