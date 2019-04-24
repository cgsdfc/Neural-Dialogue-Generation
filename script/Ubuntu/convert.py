import argparse
import os
import pickle
import operator
import logging
import numpy as np

SERBAN_UNK = '**unknown**'
JIWEILI_UNK = 'UNknown'

EOT = '__eot__'
EOU = '__eou__'
SEP = '|'

EOT_ID = 1
SUFFIX = '.txt'


def make_dict_filename(vocab):
    return 'ubuntu_{}'.format(len(vocab))


def convert_dict(input, prefix):
    logging.info('loading %s', input)
    with open(input, 'rb') as f:
        vocab_data = pickle.load(f)

    unk_tok = vocab_data[0][0]
    assert unk_tok == SERBAN_UNK
    word_getter = operator.itemgetter(0)

    output = os.path.join(prefix, make_dict_filename(vocab_data))
    with open(output, 'w') as f:
        for word in map(word_getter, vocab_data):
            print(word, file=f)
    logging.info('writen to %s', output)
    logging.info('vocab_size: %d', len(vocab_data))


def rfind(value, lst):
    return len(lst) - 1 - lst[::-1].index(value)


def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    def split_context_response(id_list: list):
        last_eot = rfind(EOT_ID, id_list)
        return id_list[:last_eot], id_list[last_eot + 1:]

    return map(split_context_response, data)


def get_length_stats(dataset):
    len_data = [(len(c), len(r)) for c, r in dataset]
    ave_len = np.mean(len_data, axis=0)
    min_len = np.min(len_data, axis=0)
    max_len = np.max(len_data, axis=0)

    print('Context', 'Response', sep='\t')
    print('Ave:', *ave_len, sep='\t')
    print('Max:', *max_len, sep='\t')
    print('Min:', *min_len, sep='\t')
    return list(map(int, ave_len))


def truncate(lst, max_len):
    if max_len is not None and max_len > 0:
        lst = lst[:max_len]
    return lst


def convert_dataset(dataset, output, context_max_len=None, response_max_len=None):
    def process(lst, max_len):
        if max_len is not None and max_len > 0:
            lst = lst[:max_len]
        lst = [id + 1 for id in lst]
        return ' '.join(map(str, lst))

    def process_pair(args):
        c, r = args
        return process(c, context_max_len), process(r, response_max_len)

    dataset = map(process_pair, dataset)

    with open(output, 'w') as f:
        for example in dataset:
            line = SEP.join(example)
            print(line, file=f)


def make_out_name(prefix, input):
    input = os.path.basename(input)
    name = input.replace('pkl', 'txt')
    return os.path.join(prefix, name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-p', '--prefix')

    # show len stats
    parser.add_argument('-l', '--len_stats', action='store_true')
    parser.add_argument('--convert-dict', action='store_true')
    parser.add_argument('--convert-data', action='store_true')
    parser.add_argument('--convert-eval', action='store_true')

    parser.add_argument('-context_max_len', type=int, default=50)
    parser.add_argument('-response_max_len', type=int, default=50)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    prefix = args.prefix or '.'

    if args.len_stats:
        get_length_stats(args.input)
    elif args.convert_dict:
        convert_dict(args.input, args.prefix)
    elif args.convert_data:
        output = make_out_name(prefix, args.input)
        dataset = list(load_dataset(args.input))
        logging.info('convert %s to %s', args.input, output)
        convert_dataset(dataset, output,
                        context_max_len=args.context_max_len,
                        response_max_len=args.response_max_len)
