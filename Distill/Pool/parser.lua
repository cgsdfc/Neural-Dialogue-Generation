local MAX_DECODED_NUM = 1000000


local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-rounds", 8, "number of distillation rounds to run")
    cmd:option("-gpu_index", 2, "")
    cmd:option("-saveFolder", "", "directory for saving output data")

    -- Dataset --
    cmd:option("-train_file", "data/t_given_s_train.txt", "initial training dataset")
    cmd:option("-dev_file", "data/t_given_s_dev.txt", "initial develop dataset")
    cmd:option("-test_file", "data/t_given_s_test.txt", "initial testing dataset")
    cmd:option("-dictPath", "data/movie_25000", "dictionary file")

    -- Attention Model --
    cmd:option("-atten_params", "", "params file for training the attention model")

    -- Decoder Model --
    cmd:option("-decoder_params", "", "params file for the decoder")
    cmd:option("-max_decoded_num", MAX_DECODED_NUM, "the maximum number of instances to Decode")

    -- Distiller --
    cmd:option("-distiller", "Encoder", "distillation method to use. Choose from {Encoder,Glove}")
    cmd:option("-save_summary", false, "whether to write a summary file per round")
    cmd:option("-batch_size", 64, "batch size for *the distiller*" ..
            "Note: it should *not* be larger than the total number of examples")
    cmd:option("-distill_rate", 0.08, "the proportion of training data to distill in a round")
    cmd:option('-freq_threshold', 100, 'frequency threshold controlling what responses are considered generic')

    -- Glove specific options --
    cmd:option("-WordMatrix", "data/wordmatrix_movie_25000", "[Glove] pretrained Glove embedding file")
    cmd:option("-distill_four_gram", false,
        "[Glove] whether to consider four-gram cooccurrence in Glove distillation")

    -- Encoder specific options --
    cmd:option("-encoder_params", "", "[Encoder] params file for the pre-trained generative model")
    cmd:option("-encoder_model", "", "[Encoder] path for loading a pre-trained generative model")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
