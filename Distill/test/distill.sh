#!/usr/bin/env bash

# The 4th step: distill the data according to the similarity score.

# Choice in (Encoder, Glove)
DISTILL_METHOD=Encoder
DISTILL_CMD=Distill/$DISTILL_METHOD/distill.lua

DISTILL_GLOVE=