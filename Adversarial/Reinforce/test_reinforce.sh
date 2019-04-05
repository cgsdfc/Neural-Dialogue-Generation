#!/usr/bin/env bash

# Test the training of the discriminator.

DISC_PARAMS=./save/test-adv-dis/params
DISC_MODEL=./save/test-adv-dis/iter6

GENERATE_PARAMS=./save/test-atten/params
GENERATE_MODEL=./save/test-atten/model8

SAVE_FOLDER=./save/test-reinforce


th Adversarial/Reinforce/train.lua \
    -disc_params ${DISC_PARAMS} \
    -disc_model ${DISC_MODEL} \
    -generate_params ${GENERATE_PARAMS} \
    -generate_model ${GENERATE_MODEL} \
    -saveFolder ${SAVE_FOLDER} \
    -gpu_index 2
