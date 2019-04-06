local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-gpu_index", 1, "the index of GPU to use")

    cmd:option("-NBest", false, "output N-best list or just a simple output")
    cmd:option("-beam_size", 7, "beam_size")
    cmd:option("-batch_size", 128, "decoding batch_size")

    cmd:option("-params_file", "", "input parameter files for a pre-trained Seq2Seq model")
    cmd:option("-model_file", "", "path for loading a pre-trained Seq2Seq model")
    cmd:option("-dictPath", "./data/movie_25000", "dictionary file")
    cmd:option("-InputFile", "./data/t_given_s_test.txt", "the input file for decoding")
    cmd:option("-OutputFile", "output.txt", "the output file to store the generated responses")

    cmd:option("-setting", "BS", "setting for decoding. Choose from sampling, BS, DiverseBS, StochasticGreedy")
    cmd:option("-StochasticGreedyNum", 1, "") -- Not found in readme.
    cmd:option("-max_length", 20, "the maximum length of a decoded response")
    cmd:option("-min_length", 0, "the minimum length of a decoded response")

    cmd:option("-max_decoded_num", 0,
        "the maximum number of instances to decode. decode the entire input set if the value is set to 0")

    cmd:option("-target_length", 0,
        "force the length of the generated target, 0 means there is no such constraints")


    cmd:option("-MMI", false, "whether to perform the mutual information reranking after decoding")

    cmd:option("-MMI_params_file", "",
        "the input parameter file for training the backward model p(s|t)")

    cmd:option("-MMI_model_file", "", "path for loading the backward model p(s|t)")

    cmd:option("-DiverseRate", 0,
        "The diverse-decoding rate for penalizing intra-sibling hypotheses in the diverse decoding model")

    cmd:option("-output_source_target_side_by_side", true,
        "output input sources and decoded targets side by side")

    cmd:option("-PrintOutIllustrationSample", false, "") -- Not found in readme.
    cmd:option("-allowUNK", false, "whether allowing to generate UNK")
    cmd:option("-onlyPred", true, "") -- Not found in readme.

    local params = cmd:parse(arg)

    -- Sanity check to avoid mysterios errors.
    assert(params.params_file ~= '', 'params_file is required')
    assert(params.model_file ~= '', 'model_file is required')

    assert(path.isfile(params.dictPath), 'dictPath must exist')
    assert(path.isfile(params.InputFile), 'input file must exist')

    if params.MMI then
        assert(path.isfile(params.MMI_model_file), 'MMI model file must exist when using MMI')
        assert(path.isfile(params.MMI_params_file), 'MMI params file must exist when using MMI')
    end

    print(params)
    return params
end

return parse_args
