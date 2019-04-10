local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-dimension", 512, "")
    cmd:option("-batch_size", 128, "")
    cmd:option("-alpha", 0.01, "")
    cmd:option("-gpu_index", 1, "the index of GPU to use")
    cmd:option("-max_iter", 8, "max number of iteration")

    cmd:option("-save_model_path", "save", "path for saving the model")
    cmd:option("-train_file", "data/t_given_s_train.txt", "")
    cmd:option("-dev_file", "data/t_given_s_dev.txt", "")
    cmd:option("-test_file", "data/t_given_s_test.txt", "")
    cmd:option("-dictPath", "data/movie_25000", "dictionary file")

    cmd:option("-forward_params_file", "Atten/save_t_given_s/params", "")
    cmd:option("-forward_model_file", "Atten/save_s_given_t/model1", "")
    cmd:option("-backward_params_file", "Atten/save_s_given_t/params", "")
    cmd:option("-backward_model_file", "Atten/save_s_given_t/model1", "")

    cmd:option("-readSequenceModel", true, "whether to load a pretrained seq2seq model")
    cmd:option("-readFutureModel", false, "whether to load a pretrained backward predictor model")
    cmd:option("-PredictorFile", "", "path for load a pretrained Soothsayer Model. does not need it at model training time")

    local params = cmd:parse(arg)
    paths.mkdir(params.save_model_path)
    print(params)
    return params
end

return parse_args
