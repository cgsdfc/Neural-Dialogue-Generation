local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-gpu_index", 1, "the index of GPU to use")

    cmd:option("-NBest", false, "output N-best list or just a simple output")
    cmd:option("-beam_size", 7, "beam_size")
    cmd:option("-batch_size", 128, "decoding batch_size")

    cmd:option("-params_file", "", "input parameter files for a pre-trained Seq2Seq model")
    cmd:option("-model_file", "", "path for loading a pre-trained Seq2Seq model")
    cmd:option("-dictPath", "data/movie_25000", "dictionary file")
    cmd:option("-InputFile", "data/t_given_s_test.txt", "the input file for decoding")
    cmd:option("-OutputFile", "output.txt", "the output file to store the generated responses")
    cmd:option('-save_params_file', '', 'if not empty, save the decoder params to this file')

    cmd:option("-setting", "BS",
        "setting for decoding. Choose from sampling, BS, DiverseBS, StochasticGreedy")

    -- sotchastic greedy sampling is a decoding technique proposed
    -- in Distill to bring in randomness and improve diversity (specificity).
    -- The model samples from the few words with the highest probability.
    -- This option controls the highest N words to sample.
    -- It is a trade-off of between sampling and greedy.
    -- The larger the value is, the more sampling is taken in.
    -- When it is 1 (default), it degrades to pure greedy.
    cmd:option("-StochasticGreedyNum", 1, "The words with the highest Num probability to sample")
    cmd:option("-max_length", 20, "the maximum length of a decoded response")
    cmd:option("-min_length", 0, "the minimum length of a decoded response")

    cmd:option("-max_decoded_num", 0,
        "the maximum number of instances to Decode. Decode the entire input set if the value is set to 0")

    cmd:option("-target_length", 0,
        "force the length of the generated target, 0 means there is no such constraints")

    cmd:option("-MMI", false, "whether to perform the mutual information reranking after decoding")
    cmd:option("-MMI_params_file", "",
        "the input parameter file for training the backward model p(s|t)")
    cmd:option("-MMI_model_file", "", "path for loading the backward model p(s|t)")

    -- Method introduced in A simple fast diverse decode algorithm.
    cmd:option("-DiverseRate", 0,
        "The diverse-decoding rate for penalizing intra-sibling hypotheses in the diverse decoding model")

    cmd:option("-output_source_target_side_by_side", true,
        "output input sources and decoded targets side by side")

    cmd:option("-PrintOutIllustrationSample", false, "print illustration sample while decoding")
    cmd:option("-allowUNK", false, "whether allowing to generate UNK")
    cmd:option("-onlyPred", true, "") -- Not found in readme.

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
