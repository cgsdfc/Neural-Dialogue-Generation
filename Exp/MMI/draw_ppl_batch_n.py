import re
import logging
import argparse
import os

PPL_RE = re.compile(r'standard perplexity: (\d+(:?\.\d+)?)')
BATCH_N_RE = re.compile(r'batch_n: (\d+)')


def parse_ppl(string):
    m = PPL_RE.search(string)
    return float(m.group(1))


def parse_batch_n(string):
    m = BATCH_N_RE.search(string)
    return int(m.group(1))


def parse_log_file(filename):
    with open(filename) as f:
        line_iter = iter(f)
        for line in line_iter:
            try:
                ppl = parse_ppl(line)
            except AttributeError:
                pass
            else:
                try:
                    next_line = next(line_iter)
                    batch_n = parse_batch_n(next_line)
                except StopIteration:
                    logging.warning('incomplete data: missing batch_n for ppl: %f', ppl)
                except AttributeError:
                    logging.error('invalid line %s', next_line)
                    raise
                yield batch_n, ppl


def write_csv(filename, data):
    with open(filename, 'w') as f:
        print('BATCH_N,PPL', file=f)
        for batch_n, ppl in data:
            print(batch_n, ppl, sep=',', file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help='logfile to be parsed')
    args = parser.parse_args()
    data = list(parse_log_file(args.logfile))

    stem, ext = os.path.splitext(args.logfile)
    filename = stem + '.csv'
    write_csv(filename, data)
