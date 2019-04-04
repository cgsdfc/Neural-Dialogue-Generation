-- Driver script to run decoding.
require "cunn"
require "cutorch"
require "nngraph"
require 'logroll'

local logger = logroll.print_logger()
local Decoder = require('./Decoder')
local parse_args = require('./parser')
local params = parse_args()

cutorch.setDevice(params.gpu_index)

logger.info('Creating Decoder...')
local decoder = Decoder.new(params)

decoder.mode = "test"
logger.info('Decoding begins...')
decoder:decode()
