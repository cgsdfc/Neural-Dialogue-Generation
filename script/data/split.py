"""
Split a dataset into train, dev and test files.
"""

import argparse
import logging
import os
from numpy.random import shuffle


def load_dataset(filename):
    logging.info('loading dataset %s', filename)
    with open(filename) as f:
        return f.readlines()


def make_name(prefix, name, dataset_filename):
    basename = os.path.basename(dataset_filename)
    stem, ext = os.path.splitext(basename)
    # file.txt => file_train.txt
    basename = stem + '_' + name + ext
    return os.path.join(prefix, basename)


def write(filename, data):
    logging.info('writing %d lines to %s', len(data), filename)
    with open(filename, 'w') as f:
        f.writelines(data)


NAMES = ('train', 'dev', 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='a text-based line-oriented dataset file')
    parser.add_argument('prefix', help='output directory')
    parser.add_argument('-train', help='training set ratio', default=0.8, type=float)
    parser.add_argument('-dev', help='develop set ratio', default=0.1, type=float)
    parser.add_argument('-test', help='test set ratio', default=0.1, type=float)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(args.prefix):
        raise ValueError('not a dir: %s', args.prefix)

    total = sum([getattr(args, key) for key in NAMES])
    if abs(total - 1) > 10e-5:
        raise ValueError('sum of all ratio is not 1')

    output_files = [make_name(args.prefix, name, args.dataset) for name in NAMES]
    dataset = load_dataset(args.dataset)
    logging.info('number of lines: %s', len(dataset))

    logging.info('shuffling dataset')
    shuffle(dataset)

    sizes = {name: round(getattr(args, name) * len(dataset)) for name in NAMES}
    logging.info('sizes: %s', sizes)

    start = 0
    for name, size in sizes.items():
        split = dataset[start:start + size]
        start += size
        filename = make_name(args.prefix, name, args.dataset)
        write(filename, split)
