local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-TrainingData", "", "path for your training data to distill")
    cmd:option("-TopResponseFile", "", "path for your extracted top frequent responses")
    cmd:option("-saveFolder", "", "directory for saving output data")

    cmd:option("-batch_size", 1280, "")
    cmd:option("-gpu_index", 1, "")
    cmd:option("-distill_rate", 0.08, "the proportion of training data to distill in this round")

    cmd:option("-params_file", "", "hyperparameters for the pre-trained generative model")
    cmd:option("-model_file", "", " path for loading a pre-trained generative model")

    cmd:option("-save_score", false, "whether to save the relevance score")
    cmd:option("-save_removed", false, "whether to save the examples got removed")

    local params = cmd:parse(arg)
    assert(path.isdir(params.saveFolder))

    params.OutputFile = path.join(params.saveFolder, path.basename(params.TrainingData))

    if params.save_score then
        params.save_score_file = path.join(params.saveFolder, 'relevance_score.txt')
    end
    if params.save_removed then
        params.save_removed_file = path.join(params.saveFolder, 'removed_examples.txt')
    end

    print(params)
    return params
end

return parse_args
