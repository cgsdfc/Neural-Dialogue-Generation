require "cunn"
require "cutorch"
require "nngraph"
require "logroll"

local logger = logroll.print_logger()
local parse_args = require("Adversarial/discriminative/parser")
local Discriminator = require("Adversarial/discriminative/Model")

local params = parse_args()
local model = Discriminator.new(params)

cutorch.manualSeed(123)
cutorch.setDevice(params.gpu_index)

logger.info('Training Adversarial Discriminator')
model:train()
logger.info('Training done.')
