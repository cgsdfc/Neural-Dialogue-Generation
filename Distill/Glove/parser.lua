local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-TrainingData", "", "")
    cmd:option("-TopResponseFile", "", "")
    cmd:option("-batch_size", 1280, "")
    cmd:option("-gpu_index", 1, "")
    cmd:option("-WordMatrix", "data/wordmatrix_movie_25000", "")
    cmd:option("-OutputFile", "", "")
    cmd:option("-encode_distill", true, "")
    cmd:option("-params_file", "", "")
    cmd:option("-model_file", "", "")
    cmd:option("-save_score", false, "")
    cmd:option("-save_score_file", "relevance_score", "")
    cmd:option("-loadscore", false, "")
    cmd:option("-total_lines", 1, "")
    cmd:option("-distill_rate", 0.08, "")
    cmd:option("-distill_four_gram", false, "")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
