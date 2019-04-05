#!/usr/bin/env bash

SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-adv-dis

th Adversarial/discriminative/train_dis.lua -gpu_index 2 -saveFolder ${SAVE_FOLDER}
