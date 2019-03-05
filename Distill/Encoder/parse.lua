local stringx = require('pl.stringx')
local cmd = torch.CmdLine()
cmd:option("-TrainingData", "", "")
cmd:option("-TopResponseFile", "", "")
cmd:option("-batch_size", 1280, "")
cmd:option("-gpu_index", 1, "")
cmd:option("-WordMatrix", "../../data/wordmatrix_movie_25000", "")
cmd:option("-OutputFile", "", "")
cmd:option("-encode_distill", true, "")
cmd:option("-params_file", "", "")
cmd:option("-model_file", "", "")

local params = cmd:parse(arg)
print(params)
return params;
