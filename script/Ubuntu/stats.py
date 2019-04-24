import argparse
import collections
import csv
import pickle
import numpy as np
import operator


def make_len_counter(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        len_of_lines = [len(example) for example in data]
    return len(data), collections.Counter(len_of_lines)


def integral(counter, n_examples):
    items = np.array(sorted(counter.items()))
    cumsum = np.cumsum(items[:, 1], axis=0)
    percent = cumsum / n_examples
    return list(zip(items[:, 0], percent))


def write_stats(filename, stats):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['len', 'count'])
        writer.writeheader()
        for len, count in sorted(stats):
            writer.writerow({'len': len, 'count': count})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    n_examples, counter = make_len_counter(args.input)
    stats = integral(counter, n_examples)
    write_stats(args.output, stats)
