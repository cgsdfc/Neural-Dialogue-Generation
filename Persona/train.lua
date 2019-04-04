require 'logroll'
require "cunn"
require "cutorch"
require "nngraph"

local logger = logroll.print_logger()
local parse_args = require("Persona/parser")
local params = parse_args()
local PersonaModel = require('Persona/Model')

cutorch.setDevice(params.gpu_index)

logger.info('Model: Persona_SpeakerAddressee')
logger.info('speakerSetting: %s', params.speakerSetting)

logger.info('Creating Model...')
local model = PersonaModel.new(params)
logger.info('model.dataset: %s', model.dataset)
model:train()
logger.info('Training done.')
