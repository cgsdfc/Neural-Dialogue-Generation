#!/usr/bin/env bash

# Run all steps of one round of the distillation procedure.
# There is no need to run many rounds since this is just *testing*.
# If we can make one round work, we already eliminate a lot of bugs.

ALL_TESTS=(
    Distill/test_steps/init.sh
    Distill/test_steps/train_seq2seq.sh
    Distill/test_steps/decode.sh
    Distill/test_steps/extract_top.sh
    Distill/test_steps/encoder_distill.sh
    Distill/test_steps/glove_distill.sh
)

ALL_MESSAGES=(
    "Copy initial dataset"
    "Train Seq2Seq-Attention model"
    "Decode a large number of responses"
    "Extract most common responses"
    "Distill using encoder"
    "Distill using Glove embeddings"
)

test ${#ALL_TESTS[@]} -eq ${#ALL_MESSAGES[@]} || exit 1

for (( i = 0; i < ${#ALL_TESTS[@]}; ++i )); do
    echo ${ALL_MESSAGES[$i]}
    . ${ALL_TESTS[$i]}
done

echo "All tests done."
