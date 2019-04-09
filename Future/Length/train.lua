require "cunn"
require "cutorch"
require "nngraph"

local parse_args = require('Future/Length/parser')
local params = parse_args()
local Model = require('Future/Length/Model')

cutorch.setDevice(params.gpu_index)
local model = Model.new(params)
model:train()
