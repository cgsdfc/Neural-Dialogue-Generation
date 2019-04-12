#!/usr/bin/env bash

DATASET=data/movie_25000
PREFIX=$(mktemp -d)

python script/data/split.py $DATASET $PREFIX
ls -l $PREFIX
#rm -rf $PREFIX
