require "torch"
require "cunn"
require "cutorch"
require "nngraph"
require 'logroll'

local parse_args = require('./parse')
local AttenModel = require("./Model")
local logger = logroll.print_logger()
local params = parse_args()

cutorch.manualSeed(123)

logger.info('Using GPU %d', params.gpu_index)
cutorch.setDevice(params.gpu_index)

local model = AttenModel.new(params)
logger.info('Created model %s', model)

logger.info('Training begins...')
model:train()
