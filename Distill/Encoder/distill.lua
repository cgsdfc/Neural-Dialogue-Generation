require "cunn"
require "cutorch"
require "nngraph"


local parse_args = require("Distill/Encoder/parser")
local Distiller = require('Distill/Encoder/Model')

print(parse_args)
local params = parse_args()
cutorch.setDevice(params.gpu_index)

local model = Distiller.new(params)
model:Distill()
