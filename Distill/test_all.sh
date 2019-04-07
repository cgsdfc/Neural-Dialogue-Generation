#!/usr/bin/env bash

# Run one round of the distillation procedure.
# There is no need to run many rounds since this is just *testing*.
# If we can make one round work, we already eliminate a lot of bugs.

ALL_TESTS=(
    Distill/test/init.sh
    Distill/test/train_seq2seq.sh
    Distill/test/decode.sh
    Distill/test/extract_top.sh
    Distill/test/encoder_distill.sh
    Distill/test/glove_distill.sh
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
