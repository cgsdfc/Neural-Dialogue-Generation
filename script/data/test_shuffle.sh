#!/usr/bin/env bash

INPUT=/home/cgsdfc/OpenSubData/dialogue_length2_6/train.txt

python script/data/shuffle.py $INPUT -d
