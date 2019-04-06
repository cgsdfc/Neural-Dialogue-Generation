#!/usr/bin/env bash

SAVE_FOLDER=./save/test-adv-dis

th Adversarial/Discriminative/train_dis.lua -gpu_index 2 -saveFolder ${SAVE_FOLDER}
