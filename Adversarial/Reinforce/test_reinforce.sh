#!/usr/bin/env bash

DISC_PARAMS=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-adv-dis/params
DISC_MODEL=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-adv-dis/iter6

GENERATE_PARAMS=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten/params
GENERATE_MODEL=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten/model8

SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-reinforce


th Adversarial/Reinforce/train.lua \
    -disc_params ${DISC_PARAMS} \
    -disc_model ${DISC_MODEL} \
    -generate_params ${GENERATE_PARAMS} \
    -generate_model ${GENERATE_MODEL} \
    -saveFolder ${SAVE_FOLDER} \
    -gpu_index 2
