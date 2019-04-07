local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-TrainingData", "", "path for your training data to distill")
    cmd:option("-TopResponseFile", "", "path for your extracted top frequent responses")
    cmd:option("-saveFolder", "", "directory for saving output data")

    cmd:option("-batch_size", 1280, "")
    cmd:option("-gpu_index", 1, "")
    cmd:option("-distill_rate", 0.08, "the proportion of training data to distill in this round")
    cmd:option("-dictPath", "data/movie_25000", "dictionary file")

    cmd:option("-params_file", "", "hyperparameters for the pre-trained generative model")
    cmd:option("-model_file", "", " path for loading a pre-trained generative model")

    cmd:option("-save_summary", false, "whether to write a summary file")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
