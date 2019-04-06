require 'logroll'
require "cunn"
require "cutorch"
require "nngraph"

local parse_args = require('Distill/Glove/parser')
local Distiller = require('Distill/Glove/Model')
local logger = logroll.print_logger()

local params = parse_args()
cutorch.setDevice(params.gpu_index)
cutorch.manualSeed(123)

local model = Distiller.new(params)

if params.save_score then
    logger.info('compute and save scores')
    model:GetScore()
else
    logger.info('distill data based on previously computed scores')
    assert(params.load_score, 'for distill, load_score must be set')
    model:Distill()
end
