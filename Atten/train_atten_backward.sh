#!/usr/bin/env bash

# Test training of the backward attention model.
# Being *backward* means predicting the *source* given the *target*.
# the dataset files should have prefix *s_given_t*.
# The backward model is needed in MMI decoding.

IMAGE=kaixhin/cuda-torch-mega:8.0
PROJECT_ROOT=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation
SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten-backward

docker run --runtime nvidia --rm  \
    -v ${PROJECT_ROOT}:${PROJECT_ROOT} \
    -w ${PROJECT_ROOT} \
    ${IMAGE} \
    th Atten/train_atten.lua -gpu_index 2 \
    -train_file ./data/s_given_t_train.txt \
    -dev_file ./data/s_given_t_train.txt \
    -test_file ./data/s_given_t_train.txt \
    -saveFolder ${SAVE_FOLDER}
