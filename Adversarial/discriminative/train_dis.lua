require "fbtorch"
require "cunn"
require "cutorch"
require "nngraph"

local parse_args = require("Adversarial/discriminative/parser")
local DisModel = require("Adversarial/discriminative/Model")

local params = parse_args()
local model = DisModel(params)

cutorch.manualSeed(123)
cutorch.setDevice(params.gpu_index)

model:train()
