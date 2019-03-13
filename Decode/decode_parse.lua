-- Define command line options and parse them.
local stringx = require('pl.stringx')
local cmd = torch.CmdLine()

cmd:option("-beam_size", 7, "beam_size")
cmd:option("-batch_size", 128, "decoding batch_size")
cmd:option("-params_file", "", "input parameter files for a pre-trained Seq2Seq model")
cmd:option("-model_file", "", "path for loading a pre-trained Seq2Seq model")
cmd:option("-setting", "BS", "setting for decoding, sampling, BS, DiverseBS,StochasticGreedy")
cmd:option("-DiverseRate", 0, "The diverse-decoding rate for penalizing intra-sibling hypotheses in the diverse decoding model")
cmd:option("-InputFile", "../data/t_given_s_test.txt", "the input file for decoding")
cmd:option("-OutputFile", "output.txt", "the output file to store the generated responses")
cmd:option("-max_length", 20, "the maximum length of a decoded response")
cmd:option("-min_length", 0, "the minimum length of a decoded response")
cmd:option("-NBest", false, "output N-best list or just a simple output")
cmd:option("-gpu_index", 1, "the index of GPU to use")
cmd:option("-allowUNK", false, "whether allowing to generate UNK")
cmd:option("-MMI", false, "whether to perform the mutual information reranking after decoding")
cmd:option("-onlyPred", true, "") -- Not found in readme.
cmd:option("-MMI_params_file", "../atten/save_s_given_t/params", "the input parameter file for training the backward model p(s|t)")
cmd:option("-MMI_model_file", "", "path for loading the backward model p(s|t)")
cmd:option("-max_decoded_num", 0, "the maximum number of instances to decode. decode the entire input set if the value is set to 0")
cmd:option("-output_source_target_side_by_side", true, "output input sources and decoded targets side by side")
cmd:option("-StochasticGreedyNum", 1, "") -- Not found in readme.
cmd:option("-target_length", 0, "force the length of the generated target, 0 means there is no such constraints")
cmd:option("-dictPath", "../data/movie_25000", "dictionary file")
cmd:option("-PrintOutIllustrationSample", false, "") -- Not found in readme.

local params = cmd:parse(arg)
print(params)

return params;
