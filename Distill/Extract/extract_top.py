from __future__ import print_function

import argparse
import collections
import logging

# Frequency >= this value is considered as generic.
FREQ_THRESHOLD = 100


def find_generic_responses(filename):
    def find(file):
        for line in file:
            yield line.split('|')[-1]

    with open(filename) as f:
        responses = list(find(f))
    logging.info('number of lines: %d', len(responses))
    counter = collections.Counter(responses)
    return [item for item in counter.items() if item[1] >= FREQ_THRESHOLD]


def load_dict(file):
    with open(file) as f:
        return {string: idx for idx, string in enumerate(f.read().splitlines())}


def to_numbers(line, dictionary):
    return [dictionary[string] for string in line.split()]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('-dictPath', default='data/movie_25000', help='dictionary file')
    args = parser.parse_args()

    dictionary = load_dict(args.dictPath)
    top_responses = find_generic_responses(args.input_file)
    logging.info('number of top responses: %d', len(top_responses))

    with open(args.output_file, 'w') as f:
        for s, freq in top_responses:
            ids = to_numbers(s, dictionary)
            ids = ' '.join(map(str, ids))
            f.write('%s|%s\n' % (ids, freq))
