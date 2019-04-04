require "torch"
require "cunn"
require "cutorch"
require "nngraph"
require 'logroll'

local parse_args = require('Atten/parser')
local AttenModel = require("Atten/Model")
local logger = logroll.print_logger()
local params = parse_args()

cutorch.manualSeed(123)

logger.info('Using GPU %d', params.gpu_index)
logger.info('Train file: %s', params.train_file)
logger.info('Test file: %s', params.test_file)
logger.info('Develop file: %s', params.dev_file)

logger.info('Output Directory: %s', params.saveFolder)
logger.info('Log file: %s', params.output_file)

cutorch.setDevice(params.gpu_index)
logger.info('Creating model...')
local model = AttenModel.new(params)
logger.info('Created model %s', model)
logger.info('Training begins...')
model:train()
