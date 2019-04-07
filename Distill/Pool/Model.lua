require 'logroll'

local MAX_DECODED_NUM = 1000000

local logger = logroll.print_logger()
local AttenModel = require('Atten/Model')
local Decoder = require('Decode/Decoder')

local DistillModelPool = torch.class('DistillModelPool')

local function get_distiller(method)
    local class = string.format('Distill/%s/Model', method)
    local parser = string.format('Distill/%s/parser', method)
    return require(class), require(parser)
end

local function mkdir_if_not_yet(dir)
    if not path.isdir(dir) then
        paths.mkdir(dir)
    end
end

-- Return subdirs for the current round.
function DistillModelPool:get_dirs_for_round(round)
    local prefix = path.join(self.saveFolder, tostring(round))
    local dirs = {
        model_dir = path.join(prefix, 'model'),
        distill_dir = path.join(prefix, 'distill'),
        tmp_dir = path.join(prefix, 'tmp'),
    }
    return dirs
end

-- Create directory structure for the current round.
function DistillModelPool:make_dirs(dirs)
    logger.info('make directories for round %d', self.round)
    for name, dir in pairs(dirs) do
        logger.info('%s: %s', name, dir)
        mkdir_if_not_yet(dir)
    end
end

function DistillModelPool:get_data_files()
    if self.round == 0 then
        return {
            train_file = self.params.train_file,
            dev_file = self.params.dev_file,
            test_file = self.params.test_file,
        }
    end
    local round_dirs = self:get_dirs_for_round(self.round - 1)
    local distill_dir = round_dirs.distill_dir
    return {
        train_file = path.join(distill_dir, self.train_file),
        dev_file = path.join(distill_dir, self.dev_file),
        test_file = path.join(distill_dir, self.test_file),
    }
end

function DistillModelPool:train_atten()
    local parse_args = require('Atten/parser')
    -- get a default params
    local params = parse_args()
    params.gpu_index = self.params.gpu_index

    -- set saveFolder
    local round_dirs = self:current_round_dirs
    local model_dir = round_dirs.model_dir
    params.saveFolder = model_dir

    local data_files = self:get_data_files()
    for k, v in pairs(data_files) do
        params[k] = v
    end


    self.current_model_params = params
    self.current_data_files = data_files
    local model = AttenModel.new(params)
    model:train()
end

function DistillModelPool:decode()
    local parse_args = require('Decode/parser')
    local params = parse_args()
    params.gpu_index = self.params.gpu_index

    local model_dir = self.current_round_dirs.model_dir
    local tmp_dir = self.current_round_dirs.tmp_dir

    params.model_file = path.join(model_dir,
        'model' .. self.current_model_params.max_iter)
    params.params_file = path.join(model_dir, 'params')
    params.InputFile = self.current_data_files.train_file
    params.OutputFile = path.join(tmp_dir, 'decode.txt')
    params.max_decoded_num = MAX_DECODED_NUM

    self.decoder_output = params.OutputFile
    local decoder = Decoder.new(params)
    decoder:decode()
end

function DistillModelPool:distill(TrainingData)
    local class, parser = get_distiller(self.params.distill_method)
    local params = parser()

    params.saveFolder = self.current_round_dirs.distill_dir
    params.distill_rate = self.params.distill_rate
    params.gpu_index = self.params.gpu_index
    params.save_summary = self.params.save_summary
    params.batch_size = self.params.batch_size
    params.TrainingData = TrainingData
    params.TopResponseFile = self.TopResponseFile

    if self.params.distill_method == 'Encoder' then
        params.params_file = self.params.params_file
        params.model_file = self.params.model_file
    end

    if self.params.distill_method == 'Glove' then
        params.WordMatrix = self.params.WordMatrix
        params.distill_four_gram = self.params.distill_four_gram
    end

    local distiller = class.new(params)
    distiller:distill()
end

function DistillModelPool:__init(params)
    self.params = params
    self.round = 0
end

function DistillModelPool:train()

end