"""
Shuffle a dataset file *in place*.
"""
import numpy as np
import argparse
import logging
import time
import tempfile
import os


def load_input(filename):
    t = time.time()
    with open(filename) as f:
        data = f.readlines()
    logging.info('load %d lines in %.4f s', len(data), time.time() - t)
    t = time.time()
    np.random.shuffle(data)
    logging.info('shuffle in %.4f s', time.time() - t)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input')
    parser.add_argument('-d', action='store_true', help='dry run')

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    data = load_input(args.input)

    if not args.d:
        _, output = tempfile.mkstemp(prefix=os.path.basename(args.input))
        logging.info('output: %s', output)
        t = time.time()
        with open(output, 'w') as f:
            f.writelines(data)
        logging.info('write done with %.4f s', time.time() - t)
        print(output)
    else:
        logging.info('dry run')
