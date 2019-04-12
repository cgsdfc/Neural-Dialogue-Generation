"""
Swap target and source.
"""
import argparse
from tqdm import tqdm

SEPARATOR = '|'


def swap_lines(file, sep):
    for line in file:
        swapped = line.strip().split(sep)[::-1]
        yield sep.join(swapped) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('swap source and target')
    parser.add_argument('input', help='input file to swap')
    parser.add_argument('output', help='output file for the swapped data')
    parser.add_argument('-sep', '-s', help='separator used to separate source and target', default=SEPARATOR)

    args = parser.parse_args()

    with open(args.input) as input_file:
        with open(args.output, 'w') as output_file:
            for line in tqdm(swap_lines(input_file, args.sep)):
                output_file.write(line)
