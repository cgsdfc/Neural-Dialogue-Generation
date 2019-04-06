local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-TrainingData", "", "path for your training data to distill")
    cmd:option("-TopResponseFile", "", "path for your extracted top frequent responses")
    cmd:option("-OutputFile", "", "file for remaining data")

    cmd:option("-batch_size", 1280, "")
    cmd:option("-gpu_index", 2, "")

    cmd:option("-WordMatrix", "data/wordmatrix_movie_25000", "pretrained Glove embedding file")

    cmd:option("-save_score", false, "whether to save relevance scores")
    cmd:option("-load_score", false, "whether to load already-computed relevance scores")

    cmd:option("-save_score_file", "relevance_score",
        "path for saving relevance_score for each instance in the training set")

    cmd:option("-total_lines", 1, "number of lines in your TrainingData")

    cmd:option("-distill_rate", 0.08,
        "the proportion of training data to distill in this round")

    cmd:option("-distill_four_gram", false,
        "whether to remove all training instances that share four-grams with any one of the top frequent responses")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
