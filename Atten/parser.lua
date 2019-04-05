require 'logroll'

local logger = logroll.print_logger()

local function parse_args()
    local cmd = torch.CmdLine()

    cmd:option("-batch_size", 128, "batch size")
    cmd:option("-dimension", 512, "vector dimensionality")
    cmd:option("-dropout", 0.2, "dropout rate")

    cmd:option("-train_file", "./data/t_given_s_train.txt", "")
    cmd:option("-dev_file", "./data/t_given_s_dev.txt", "")
    cmd:option("-test_file", "./data/t_given_s_test.txt", "")

    cmd:option("-init_weight", 0.1, "weight used in random uniform initializer as [-init_weight, init_weight]")
    cmd:option("-alpha", 1, "initial learning rate")
    cmd:option("-start_halve", 6, "when to start halving the learning rate. Set to -1 to disable the halving")
    cmd:option("-max_length", 100, "");
    cmd:option("-vocab_source", 25010, "source vocabulary size")
    cmd:option("-vocab_target", 25010, "target vocabulary size")
    cmd:option("-thres", 5, "gradient clipping thres")
    cmd:option("-max_iter", 8, "max number of iteration")
    cmd:option("-source_max_length", 50, "max length of source sentences")
    cmd:option("-target_max_length", 50, "max length of target sentences")
    cmd:option("-layers", 2, "number of lstm layers")
    cmd:option("-saveFolder", "save", "the folder to save models and parameters")
    cmd:option("-reverse", false, "whether to reverse the sources")
    cmd:option("-gpu_index", 1, "the index of GPU to use")
    cmd:option("-saveModel", true, "whether to save the trained model")
    cmd:option("-dictPath", "./data/movie_25000", "dictionary file")

    logger.info('Parsing cmdline arguments...')
    local params = cmd:parse(arg)

    params.save_prefix = path.join(params.saveFolder, "model")
    params.save_params_file = path.join(params.saveFolder, "params")

    if not path.isdir(params.saveFolder) then
        logger.info('mkdir %s', params.saveFolder)
        paths.mkdir(params.saveFolder)
    end

    return params;
end

return parse_args
