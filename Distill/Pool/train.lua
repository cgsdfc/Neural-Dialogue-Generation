require 'cutorch'

local parse_args = require('Distill/Pool/parser')
local Model = require('Distill/Pool/Model')

local params = parse_args()
cutorch.setDevice(params.gpu_index)

local model = Model.new(params)
model:train()
