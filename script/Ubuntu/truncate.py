import argparse
import os


def truncate_file(filename, output, max_len):
    def truncate(line):
        words = line.split()
        return words[:max_len]

    with open(filename) as f, open(output, 'w') as out:
        for line in f:
            line = ' '.join(truncate(line))
            print(line, file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-m', '--max_len', default=50)
    args = parser.parse_args()

    output = os.path.join(args.prefix, os.path.basename(args.input))
    truncate_file(args.input, output, args.max_len)
