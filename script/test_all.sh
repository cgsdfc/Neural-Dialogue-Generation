#!/usr/bin/env bash

# This script runs all the test scripts from all modules, sequentially.

ALL_TEST_SCRIPTS=(
    Atten/test_train.sh
    Atten/test_train_forward.sh
    Atten/test_train_backward.sh

    Decode/test_decode.sh
    Decode/test_decode_MMI.sh

    Persona/test_persona_speaker.sh
    Persona/test_persona_speaker_addressee.sh

    Adversarial/Discriminative/test_train_dis.sh
    Adversarial/Reinforce/test_reinforce.sh

    Distill/test_steps/test_all.sh
    Distill/Pool/test_pool.sh

    Future/Length/test_length.sh
    Future/Backward/test_backward.sh
    Future/Decode/test_decode_backward.sh
    Future/Decode/test_decode_length.sh
)

for test in ${ALL_TEST_SCRIPTS[@]}
do
    echo "Run: $test"
    . ${test}
done
