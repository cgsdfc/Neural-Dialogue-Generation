local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-TrainingData", "", "path for your training data to distill")
    cmd:option("-TopResponseFile", "", "path for your extracted top frequent responses")
    cmd:option("-batch_size", 1280, "")
    cmd:option("-gpu_index", 1, "")
    cmd:option("-WordMatrix", "data/wordmatrix_movie_25000", "")
    cmd:option("-OutputFile", "", "")
    cmd:option("-encode_distill", true, "")
    cmd:option("-params_file", "", "hyperparameters for the pre-trained generative model")
    cmd:option("-model_file", "", " path for loading a pre-trained generative model")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
