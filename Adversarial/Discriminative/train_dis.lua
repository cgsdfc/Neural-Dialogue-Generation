require "cunn"
require "cutorch"
require "nngraph"
require "logroll"

local logger = logroll.print_logger()
local parse_args = require("Adversarial/Discriminative/parser")
local DisModel = require("Adversarial/Discriminative/Model")

local params = parse_args()

-- *Note*: Must set device before creating a model!!!
-- Especially when GPU-1 is fully loaded!
logger.info('Using GUP %d', params.gpu_index)
cutorch.setDevice(params.gpu_index)

cutorch.manualSeed(123)

logger.info('Training DisModel')
logger.info('Creating DisModel...')
local model = DisModel.new(params)

logger.info('Training begin...')
model:train()
logger.info('Training done.')
