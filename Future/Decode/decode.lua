require "cunn"
require "cutorch"
require "nngraph"

local parse_args = require('Future/Decode/parser')
local params = parse_args()
cutorch.setDevice(params.gpu_index)

local Model = require('Future/Decode/Model')
local decode_model = Model.new(params)

decode_model.mode = "test"
decode_model:decode()
