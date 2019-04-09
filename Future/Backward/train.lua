require "cunn"
require "cutorch"
require "nngraph"

local parse_args = require('Future/Backward/parser')
local params = parse_args()
local Model = require('Future/Backward/Model')

cutorch.setDevice(params.gpu_index)
local model = Model.new(params)
model:train()
