require "cunn"
require "cutorch"
require "nngraph"
require 'logroll'

local logger = logroll.print_logger()
local RLModel = require('Adversarial/Reinforce/RLModel')
local parse_args = require('Adversarial/Reinforce/parser')

local params = parse_args()
cutorch.manualSeed(123)
cutorch.setDevice(params.gpu_index)

logger.info('To train DisModel %d times', params.dSteps)
logger.info('To train GenModel %d times', params.gSteps)

logger.info('Creating RLModel...')
local model = RLModel.new(params)

logger.info('Training begins...')
model:train()
logger.info('Training done.')
