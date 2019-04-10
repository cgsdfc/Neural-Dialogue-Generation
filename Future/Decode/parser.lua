local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-beam_size", 7, "beam_size")
    cmd:option("-batch_size", 128, "decoding batch_size")
    cmd:option("-gpu_index", 1, "the index of GPU to use")

    cmd:option("-params_file", "Atten/save_t_given_s/params", "params of a pretrained seq2seq model")
    cmd:option("-model_file", "Atten/save_t_given_s/model1", "weights of a pretrained seq2seq model")
    cmd:option("-dictPath", "data/movie_25000", "dictionary file")

    cmd:option("-InputFile", "", "input message file")
    cmd:option("-OutputFile", "", "output response file")

    cmd:option("-setting", "BS", "setting for decoding. {sampling,BS,DiverseBS,StochasticGreedy}")

    -- Simple fast diverse --
    cmd:option("-DiverseRate", 0,
        "The diverse-decoding rate for penalizing intra-sibling hypotheses in the diverse decoding model")

    -- MMI Reranking --
    cmd:option("-MMI", false, "whether to perform the mutual information reranking after decoding")
    cmd:option("-MMI_params_file", "", "the input parameter file for training the backward model p(s|t)")
    cmd:option("-MMI_model_file", "", "path for loading the backward model p(s|t)")

    cmd:option("-max_length", 0, "the maximum length of a decoded response")
    cmd:option("-min_length", 0, "the minimum length of a decoded response")
    cmd:option("-NBest", true, "output N-best list or just a simple output")
    cmd:option("-allowUNK", true, "whether allowing to generate UNK")
    cmd:option("-onlyPred", true, "")

    cmd:option("-max_decoded_num", 0,
        "the maximum number of instances to Decode." ..
                " Decode the entire input set if the value is set to 0")
    cmd:option("-output_source_target_side_by_side", false,
        "output input sources and decoded targets side by side")

    -- StochasticGreedy --
    cmd:option("-StochasticGreedyNum", 1, "")

    -- Future Prediction --
    cmd:option("-Task", "", "{length,backward}")

    -- LenModel --
    cmd:option("-target_length", 0,
        "force the length of the generated target. Required for Task == length." ..
                "0 means there is no such constraints")

    -- The regression model trained with LenModel or BackwardModel (MMIModel)
    cmd:option("-FuturePredictorModelFile", "",
        "path for loading a pre-trained Soothsayer future prediction model." ..
                "It should be a either length or backward model.")

    -- The lambda hyparam in the paper. The weight of the predictor output in the probability
    -- of the words. The larger the value is, the more influence the predictor will gain --
    -- target length gets closer to pre-specified length -- backward probability gets closer to
    -- ground truth. The default is 0, which disables the future predictor totally.
    -- To actually use future prediction, set it to 1 or larger like 5.
    cmd:option("-PredictorWeight", 0, "the weight for the Soothsayer model")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
