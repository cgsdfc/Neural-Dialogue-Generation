#!/usr/bin/env bash

IMAGE=kaixhin/cuda-torch-mega:8.0
PROJECT_ROOT=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation
SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-forward

docker run --runtime nvidia --rm  \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    -w ${PROJECT_ROOT} \
    ${IMAGE} \
    th Atten/train_atten.lua -gpu_index 2 \
    -train_file ./data/t_given_s_train.txt \
    -dev_file ./data/t_given_s_dev.txt \
    -test_file ./data/t_given_s_test.txt \
    -saveFolder ${SAVE_FOLDER}
