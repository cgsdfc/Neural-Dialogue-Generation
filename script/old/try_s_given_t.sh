#!/usr/bin/env bash

th train.lua -train_file \
    ./data/s_given_t_train.txt \
    -dev_file ./data/s_given_t_train.txt \
    -test_file ./data/s_given_t_train.txt \
    -saveFolder save_s_given_t
