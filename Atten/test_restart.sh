#!/usr/bin/env bash

SAVE_FOLDER=save/test-atten

th Atten/train.lua \
    -gpu_index 1 \
    -saveFolder ${SAVE_FOLDER} \
    -batch_size 10 \
    -restart
