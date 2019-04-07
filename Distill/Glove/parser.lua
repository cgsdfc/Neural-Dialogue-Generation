local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-TrainingData", "", "path for your training data to distill")
    cmd:option("-TopResponseFile", "", "path for your extracted top frequent responses")
    cmd:option("-saveFolder", "", "directory for saving output data")

    cmd:option("-batch_size", 1280, "")
    cmd:option("-gpu_index", 2, "")

    cmd:option("-WordMatrix", "data/wordmatrix_movie_25000", "pretrained Glove embedding file")

    cmd:option("-distill_rate", 0.08, "the proportion of training data to distill in this round")

    cmd:option("-distill_four_gram", false,
        "whether to remove all training instances that share four-grams with any one of the top frequent responses")
    cmd:option("-save_summary", false, "whether to write a summary file")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
