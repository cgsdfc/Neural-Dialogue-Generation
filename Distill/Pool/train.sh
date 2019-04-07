#!/usr/bin/env bash

ATTEN_PARAMS=save/test-atten/params
DECODER_PARAMS=save/test-decode/params
BATCH_SIZE=1
FREQ_THRES=5
ROUNDS=8

ENCODER_PARAMS=save/test-atten/params
ENCODER_MODEL=save/test-atten/model8

SAVE_FOLDER=save/test-pool/
DISTILLER=Glove
#    -encoder_params $ENCODER_PARAMS \
#    -encoder_model $ENCODER_MODEL \
# The Encoder is currently broken

th Distill/Pool/train.lua \
    -atten_params $ATTEN_PARAMS \
    -decoder_params $DECODER_PARAMS \
    -gpu_index 2 \
    -batch_size $BATCH_SIZE \
    -saveFolder $SAVE_FOLDER \
    -distiller $DISTILLER \
    -freq_threshold $FREQ_THRES \
    -rounds $ROUNDS \
    -save_summary
