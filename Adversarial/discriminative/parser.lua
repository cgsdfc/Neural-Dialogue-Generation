local function parse_args()
    local cmd = torch.CmdLine()
    cmd:option("-batch_size", 128, "batch size")
    cmd:option("-dimension", 512, "vector dimensionality")
    cmd:option("-dropout", 0.2, "dropout rate")

    -- Positive and Negative datasets for train, dev and test.
    cmd:option("-pos_train_file", "data/t_given_s_train.txt",
        "human generated training examples (positive example)")

    cmd:option("-neg_train_file", "data/decoded_train.txt",
        "machine generated training examples (negative example)")

    cmd:option("-pos_dev_file", "data/t_given_s_dev.txt",
        "human generated dev examples (positive example)")

    cmd:option("-neg_dev_file", "data/decoded_dev.txt",
        "machine generated dev examples (negative example)")

    cmd:option("-pos_test_file", "data/t_given_s_test.txt",
        "human generated test examples (positive example)")

    cmd:option("-neg_test_file", "data/decoded_test.txt",
        "machine generated test examples (negative example)")

    cmd:option("-source_max_length", 50, "maximum sequence length")

    cmd:option("-init_weight", 0.1, "")
    cmd:option("-alpha", 0.01, "")
    cmd:option("-start_halve", 6, "")
    cmd:option("-max_length", 100, "")
    cmd:option("-vocab_size", 25010, "")
    cmd:option("-thres", 5, "gradient clipping thres")
    cmd:option("-max_iter", 6, "max number of iteration")
    cmd:option("-layers", 1, "")

    cmd:option("-dialogue_length", 2,
        "the number of turns for a dialogue. the model supports multi-turn dialgoue classification")

    cmd:option("-saveFolder", "save/", "the folder to save models and parameters")

    cmd:option("-gpu_index", 2, "which GPU to use")

    cmd:option("-saveModel", true, "whether to save the model")

    local params = cmd:parse(arg)
    print(params)
    return params
end

return parse_args
