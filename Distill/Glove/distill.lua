require "cunn"
require "cutorch"
require "nngraph"

local parse_args = require('Distill/Glove/parser')
local Distiller = require('Distill/Glove/Model')

local params = parse_args()
cutorch.setDevice(params.gpu_index)
cutorch.manualSeed(123)

local model = Distiller.new(params)

if params.save_score then
    model:GetScore()
else
    model:Distill()
end
