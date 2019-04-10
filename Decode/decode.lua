require "cunn"
require "cutorch"
require "nngraph"
require 'logroll'

local logger = logroll.print_logger()
local Decoder = require('Decode/Decoder')
local parse_args = require('Decode/parser')
local params = parse_args()

cutorch.setDevice(params.gpu_index)

if params.MMI then
    logger.info('Using MMI decoding')
end

logger.info('Creating Decoder...')
local decoder = Decoder.new(params)

decoder.mode = "test"
logger.info('Decoding begins...')
decoder:decode()
logger.info('Decoding done.')
