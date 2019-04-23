import numpy as np
import argparse
import logging
import time

SUFFIX = '_shuffle.txt'


def load_input(filename):
    t = time.time()
    with open(filename) as f:
        data = f.readlines()
    logging.info('load %d lines in %.4f s', len(data), time.time() - t)
    t = time.time()
    np.random.shuffle(data)
    logging.info('shuffle in %.4f s', time.time() - t)
    return data


def get_output(filename):
    return filename + SUFFIX


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-d', action='store_true', help='dry run')
    args = parser.parse_args()
    if args.v:
        logging.basicConfig(level=logging.INFO)
    data = load_input(args.input)

    if not args.d:
        output = get_output(args.input)
        t = time.time()
        with open(output, 'w') as f:
            f.writelines(data)
        logging.info('write to %s in %.4f s', output, time.time() - t)
        print(output)
