"""
A wrapper script to run train_atten.py with hparams fixed to those in the paper
diversity-promoting.
"""
import argparse
import os
import tempfile

project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default=tempfile.mkdtemp(), help="folder to save models and parameters")
parser.add_argument("--train_file", default=os.path.join(data_dir, "t_given_s_train.txt"),
                    help="training dataset")
parser.add_argument("--dev_file", default=os.path.join(data_dir, "t_given_s_dev.txt"),
                    help="develop dataset")
parser.add_argument("--test_file", default=os.path.join(data_dir, "t_given_s_test.txt"),
                    help="test dataset")
parser.add_argument("--vocab_file", default=os.path.join(data_dir, "movie_25000"),
                    help="dictionary/vocabulary file")

# Hyper-parameters from the paper.
hparams_config = {
    "batch_size": 256,
    "dimension": 1000,
    # dropout is unknown, use default
    "init_weight": 0.08,
    "alpha": 0.1,  # learning rate.
    "start_halve": -1,  # Never half the lr.
    # max_length: use default.
    # vocab_source, vocab_target: use default.
    "thres": 1,  # Gradient clipping threshold.
    # max_iter unknown, use default.
    # source_max_length, target_max_length: use default.
    "layers": 4,
    # reverse: false
    # saveModel: true
}

# Build the argument list.
lua_train_script = os.path.join(project_root, "Atten", "train_atten.lua")
hparams_options = ['th', lua_train_script]

for key, val in hparams_config.items():
    hparams_options.append("-" + key)
    hparams_options.append(str(val))

renamed_options = {
    "out_dir": "saveFolder",
    "vocab_file": "dictPath",
}

# logging is already done by lua code.
args = parser.parse_args()

for key, val in vars(args).items():
    hparams_options.append("-" + renamed_options.get(key, key))
    hparams_options.append(str(val))

del key, val

import subprocess
import sys

print("Invoking", " ".join(hparams_options), file=sys.stderr)
sys.exit(subprocess.call(hparams_options))
