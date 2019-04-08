local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-params_file", "Atten/save_t_given_s/params", "")
    cmd:option("-model_file", "Atten/save_t_given_s/model1", "")
    cmd:option("-batch_size", 128, "")
    cmd:option("-gpu_index", 1, "")
    cmd:option("-save_model_path", "save", "")
    cmd:option("-alpha", 0.0001, "")
    cmd:option("-train_file", "data/t_given_s_train.txt", "")
    cmd:option("-dev_file", "data/t_given_s_dev.txt", "")
    cmd:option("-test_file", "data/t_given_s_test.txt", "")
    cmd:option("-gpu_index", 1, "the index of GPU to use")
    cmd:option("-readSequenceModel", true, "whether to load a pretrained seq2seq model")
    cmd:option("-readFutureModel", false, "whether to load a pretrained lenght predictor model")
    cmd:option("-FuturePredictorModelFile", "", "")

    local params = cmd:parse(arg)
    paths.mkdir(params.save_model_path)
    print(params)
    return params
end

return parse_args
