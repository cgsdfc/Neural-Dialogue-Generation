#!/usr/bin/env bash

# Test the training of the speaker_addressee model.

SAVE_FOLDER=/home/cgsdfc/deployment/Models/Dialogue/Neural-Dialogue-Generation/save/test-persona-speaker-addressee

th Persona/train.lua -saveFolder ${SAVE_FOLDER} -speakerSetting speaker_addressee
