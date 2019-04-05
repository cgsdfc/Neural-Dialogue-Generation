#!/usr/bin/env bash

# Test training of the attention model.
SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-atten

th Atten/train_atten.lua -gpu_index 2 \
    -saveFolder ${SAVE_FOLDER}
