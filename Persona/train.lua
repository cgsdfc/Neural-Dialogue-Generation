require 'logroll'
require "cunn"
require "cutorch"
require "nngraph"

local logger = logroll.print_logger()
local parse_args = require("Persona/parser")
local params = parse_args()
local PersonaModel = require('Persona/Model')

cutorch.setDevice(params.gpu_index)

logger.info('speakerSetting: %s', params.speakerSetting)

logger.info('Creating Model...')
local model = PersonaModel.new(params)
logger.info('Using dataset class: %s', model.dataset)

logger.info('Training begins...')
model:train()
logger.info('Training done.')
