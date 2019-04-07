require 'logroll'

local logger = logroll.print_logger()
local MAX_DECODED_NUM = 1000000

local function check_params(params)
    logger.info('Checking viability of initial dataset')
    local data_files = {
        'train_file',
        'dev_file',
        'test_file',
    }
    for _, file in ipairs(data_files) do
        local filename = params[file]
        assert(path.isfile(filename), file .. ' does not exist!')
    end
    logger.info('OK')


    local known_methods = { Encoder = 1, Glove = 1 }
    if not known_methods[params.distiller] then
        error('Unknown distiller: ' .. params.distiller)
    end
    logger.info('Using distiller: %s', params.distiller)

    if params.distiller == 'Encoder' then
        logger.info('Checking viability of pretrained encoder')
        assert(path.isfile(params.params_file), 'params_file does not exist!')
        assert(path.isfile(params.model_file), 'model_file does not exist!')
        logger.info('OK')
    end

    if params.distiller == 'Glove' then
        logger.info('Checking viability of Glove embeddings')
        assert(path.isfile(params.WordMatrix), 'WordMatrix does not exist!')
        logger.info('OK')
    end

    if not (params.rounds > 0) then
        error('rounds must be positive: ' .. params.rounds)
    end
end

local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-rounds", 8, "number of distillation rounds to run")

    cmd:option("-train_file", "data/t_given_s_train.txt", "")
    cmd:option("-dev_file", "data/t_given_s_dev.txt", "")
    cmd:option("-test_file", "data/t_given_s_test.txt", "")
    cmd:option("-dictPath", "data/movie_25000", "dictionary file")

    cmd:option("-atten_params", "", "params file for training the attention model")
    cmd:option("-decoder_params", "", "params file for the decoder")
    cmd:option("-max_decoded_num", MAX_DECODED_NUM, "the maximum number of instances to decode")

    cmd:option("-distiller", "Encoder", "distillation method to use. Choose from {Encoder,Glove}")

    cmd:option("-save_summary", false, "whether to write a summary file per round")

    cmd:option("-batch_size", 64, "number of examples to be processed at one operation." ..
            "Note: it should *not* be larger than the total number of all examples")

    cmd:option("distill_rate", 0.08, "the proportion of training data to distill in a round")
    cmd:option("-gpu_index", 2, "")
    cmd:option("-saveFolder", "", "directory for saving output data")

    -- Glove specific options:
    cmd:option("-WordMatrix", "data/wordmatrix_movie_25000", "[Glove] pretrained Glove embedding file")
    cmd:option("-distill_four_gram", false,
        "[Glove] whether to consider four-gram cooccurrence in Glove distillation")

    -- Encoder specific options:
    cmd:option("-encoder_params", "", "[Encoder] params file for the pre-trained generative model")
    cmd:option("-encoder_model", "", "[Encoder] path for loading a pre-trained generative model")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
