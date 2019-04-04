#!/usr/bin/env bash

IMAGE=kaixhin/cuda-torch-mega:8.0
PROJECT_ROOT=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation
SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test

docker run --runtime nvidia --rm  \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    -w ${PROJECT_ROOT} \
    ${IMAGE} \
    th Atten/train_atten.lua -gpu_index 2 \
    -saveFolder ${SAVE_FOLDER}
