local function parse_args()
    local cmd = torch.CmdLine()

    cmd:option("-batch_size", 128, "")
    cmd:option("-alpha", 0.0001, "learning rate")
    cmd:option("-max_iter", 8, "max number of iteration")
    cmd:option("-gpu_index", 1, "the index of GPU to use")

    cmd:option("-params_file", "Atten/save_t_given_s/params", "load hyperparameters for a pre-trained generative model")
    cmd:option("-model_file", "Atten/save_t_given_s/model1", "path for loading the pre-trained generative model")
    cmd:option("-train_file", "data/t_given_s_train.txt", "")
    cmd:option("-dev_file", "data/t_given_s_dev.txt", "")
    cmd:option("-test_file", "data/t_given_s_test.txt", "")
    cmd:option("-save_model_path", "save", "path for saving the model")

    cmd:option("-readSequenceModel", true,
        "whether to read a pretrained seq2seq model. this variable has to be set to true when training the model")
    cmd:option("-readFutureModel", false,
        "whether to load a pretrained Soothsayer Model. this variable has to be set to false when training the model")
    cmd:option("-FuturePredictorModelFile", "", "path for load a pretrained Soothsayer Model. does not need it at model training time")

    local params = cmd:parse(arg)
    paths.mkdir(params.save_model_path)
    print(params)
    return params
end

return parse_args
