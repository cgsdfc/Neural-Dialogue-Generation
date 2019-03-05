local stringx = require('pl.stringx')
local cmd = torch.CmdLine()
cmd:option("-batch_size", 128, "batch size")
cmd:option("-dimension", 512, "vector dimensionality")
cmd:option("-dropout", 0.2, "dropout rate")
cmd:option("-pos_train_file", "../../data/t_given_s_train.txt", "")
cmd:option("-neg_train_file", "../../data/decoded_train.txt", "")
cmd:option("-pos_dev_file", "../../data/t_given_s_dev.txt", "")
cmd:option("-neg_dev_file", "../../data/decoded_dev.txt", "")
cmd:option("-pos_test_file", "../../data/t_given_s_test.txt", "")
cmd:option("-neg_test_file", "../../data/decoded_test.txt", "")
cmd:option("-source_max_length", 50, "")
cmd:option("-init_weight", 0.1, "")
cmd:option("-alpha", 0.01, "")
cmd:option("-start_halve", 6, "")
cmd:option("-max_length", 100, "");
cmd:option("-vocab_size", 25010, "")
cmd:option("-thres", 5, "gradient clipping thres")
cmd:option("-max_iter", 6, "max number of iteration")
cmd:option("-layers", 1, "")
cmd:option("-dialogue_length", 2, "")
cmd:option("-save_model_path", "save", "")
cmd:option("-save_params_file", "save/params", "")
cmd:option("-output_file", "", "")
cmd:option("-gpu_index", 1, "")
cmd:option("-saveModel", true, "")
local params = cmd:parse(arg)
paths.mkdir(params.save_model_path)
print(params)
return params;
