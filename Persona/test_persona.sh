#!/usr/bin/env bash

# Test the training of the default persona model.

SAVE_FOLDER=./save/test-persona

th Persona/train.lua -saveFolder ${SAVE_FOLDER}
