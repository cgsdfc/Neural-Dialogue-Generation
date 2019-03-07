import argparse
import os


def parse_args(args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-batch_size", type=int, default=128, help="batch size")
    parser.add_argument("-dimension", type=int, default=512, help="vector dimensionality")
    parser.add_argument("-dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("-train_file", default="../data/t_given_s_train.txt")
    parser.add_argument("-dev_file", default="../data/t_given_s_dev.txt")
    parser.add_argument("-test_file", default="../data/t_given_s_test.txt")
    parser.add_argument("-init_weight", type=float, default=0.1)
    parser.add_argument("-alpha", default=1, type=float)
    parser.add_argument("-start_halve", default=6, type=int)
    parser.add_argument("-max_length", default=100, type=int)
    parser.add_argument("-vocab_source", default=25010, type=int)
    parser.add_argument("-vocab_target", default=25010, type=int)
    parser.add_argument("-thres", default=5, type=int, help="gradient clipping thres")
    parser.add_argument("-max_iter", default=8, type=int, help="max number of iteration")
    parser.add_argument("-source_max_length", default=50)
    parser.add_argument("-layers", default=2, type=int)
    parser.add_argument("-saveFolder", default="save")
    parser.add_argument("-reverse", action="store_true")
    parser.add_argument("-gpu_index", default=1, type=int, help="the index of GPU to use")
    parser.add_argument("-saveModel", action="store_false")
    parser.add_argument("-dictPath", default="../data/movie_25000")

    args = parser.parse_args(args)
    args.save_prefix = os.path.join(args.saveFolder, "model")
    args.save_params_file = os.path.join(args.saveFolder, "params")
    args.output_file = os.path.join(args.saveFolder, "log")
    os.mkdir(args.saveFolder)
    print(args)
    return args


if __name__ == '__main__':
    parse_args()
