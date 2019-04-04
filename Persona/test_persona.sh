#!/usr/bin/env bash

SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-persona

th Persona/train.lua -saveFolder ${SAVE_FOLDER}
