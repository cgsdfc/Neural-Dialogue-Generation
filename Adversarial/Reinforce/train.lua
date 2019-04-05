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

logger.info('dSteps: %d', params.dSteps)
logger.info('gSteps: %d', params.gSteps)
logger.info('batch_size: %d', params.batch_size)

local num_batches_per_epoch = params.batch_size * (params.dSteps + params.gSteps)
logger.info('num_batches_per_epoch: %d', num_batches_per_epoch)

logger.info('TeacherForce: %s', params.TeacherForce)
logger.info('trainData: %s', params.trainData)

logger.info('Creating RLModel...')
local model = RLModel.new(params)

logger.info('Training begins...')
model:train()
logger.info('Training done.')
