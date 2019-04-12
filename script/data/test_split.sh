#!/usr/bin/env bash

# test if we split things correctly.

DATASET=data/movie_25000
PREFIX=$(mktemp -d)

python script/data/split.py $DATASET $PREFIX
ls -l $PREFIX
rm -rf $PREFIX
