-- The implementation makes use of a *symbolic directory structure* to run the system.
-- The symbolic directory is a nested table holded in memory and backed by physical directories.
-- The keys of the table represent the symbolic dirs or files while the value of the table represents
-- phisical directories or files. The symbolic structure is the public visible API and is rather ideal
-- and straightforward while the phisical structure is hidden aways and rapiddly changing.
--
-- The symbolic structure is:
-- * round_dir
--      * data_dir
--          * train_file
--          * dev_file
--          * test_file
--      * model_dir
--          * params_file
--          * model_file
--      * tmp_dir
--          * decoder_output
--          * top_response
--      * distill
--          * train_file
--          * dev_file
--          * test_file
--
-- This structure describes all things required to run the system.
-- More or less physical files will be produced.
-- All the round_dirs for all rounds are kept in the memory in the whole training.
-- Specially, round-1, which is the initial round, maps its data_dir and underlying files
-- to the initial data files passed in. For rounds after round-1, data_dir is linked to the distill_dir
-- of the previous round.


require 'logroll'


local logger = logroll.print_logger()
local extract_top = require('Distill/Extract/extract_top')
local AttenModel = require('Atten/Model')
local Decoder = require('Decode/Decoder')

local DistillModelPool = torch.class('DistillModelPool')

local function mkdir_if_not_yet(dir)
    if not path.isdir(dir) then
        paths.mkdir(dir)
    end
end

function DistillModelPool:train_atten()
    local params = self.atten_params
    local round_dir = self.round_dir
    local data_dir = round_dir.data_dir

    for key, value in pairs(data_dir) do
        params[key] = value
    end
    params.saveFolder = round_dir.model_dir.dir
    local model = AttenModel.new(params)
    model:train()
end

function DistillModelPool:decode()
    local params = self.decoder_params
    local round_dir = self.round_dir

    params.model_file = round_dir.model_dir.model_file
    params.params_file = round_dir.model_dir.param_file
    params.InputFile = round_dir.data_dir.train_file
    params.OutputFile = round_dir.tmp_dir.decoder_output

    local decoder = Decoder.new(params)
    decoder:decode()
end

function DistillModelPool:extract_top()
    local dict_file = self.dict_file
    local input = self.round_dir.tmp_dir.decoder_output
    local output = self.round_dir.tmp_dir.top_response
    extract_top(input, output, dict_file)
end

function DistillModelPool:distill()
    local class = self.distiller_class
    local params = self.distill_params
    local round_dir = self.round_dir

    params.saveFolder = round_dir.distill_dir.dir
    params.TopResponseFile = round_dir.tmp_dir.top_response

    for _, data_file in ipairs(round_dir.data_dir) do
        params.TrainingData = data_file
        local distiller = class.new(params)
        distiller:distill()
    end
end

local function load_object(filename)
    local file = torch.DiskFile(filename):binary()
    local obj = file:readObject()
    file:close()
    return obj
end

-- Load and modify the params for the attention model
function DistillModelPool:load_atten_params(params)
    local atten_params = load_object(params.atten_params)
    atten_params.gpu_index = params.gpu_index
    atten_params.saveModel = true
    return atten_params
end


function DistillModelPool:load_decoder_params(params)
    local decoder_params = load_object(params.decoder_params)
    decoder_params.gpu_index = params.gpu_index

    if decoder_params.NBest then
        logger.warn('decoder should not set NBest and should produce a single ouput per input')
        decoder_params.NBest = false
    end

    decoder_params.max_decoded_num = params.max_decoded_num
    return decoder_params
end

function DistillModelPool:load_distiller_params(params)
    local distiller = params.distiller
    self.distiller_class = require(string.format('Distill/%s/Model', distiller))
    local distiller_params = {
        distill_rate = params.distill_rate,
        save_summary = params.save_summary,
        batch_size = params.batch_size,
        gpu_index = params.gpu_index,
    }

    if distiller == 'Encoder' then
        distiller_params.params_file = params.encoder_params
        distiller_params.model_file = params.encoder_model
    end

    if distiller == 'Glove' then
        distiller_params.WordMatrix = params.WordMatrix
        distiller_params.distill_four_gram = params.distill_four_gram
    end

    return distiller_params
end

-- Walk the symbolic tree and create necessary dirs.
function DistillModelPool:create_physical_dirs()
    for _, dir in ipairs(self.round_dir) do
        local first_file = dir[1]
        assert(first_file ~= nil, 'there must be at least one file!')
        dir = path.dirname(first_file)
        mkdir_if_not_yet(dir)
    end
end

function DistillModelPool:create_symbolic_dirs()
    local function template(round, params, atten_params)
        local phy_round_dir = path.join(params.saveFolder, round)
        local phy_model_dir = path.join(phy_round_dir, 'model')
        local phy_distill_dir = path.join(phy_round_dir, 'distill')
        local phy_tmp_dir = path.join(phy_round_dir, 'tmp')

        local max_iter = atten_params.max_iter

        local train_file = path.basename(params.train_file)
        local dev_file = path.basename(params.dev_file)
        local test_file = path.basename(params.test_file)

        return {
            data_dir = {
                train_file = '',
                dev_file = '',
                test_file = '',
            },
            model_dir = {
                dir = phy_model_dir, -- point to itself.
                params_file = path.join(phy_model_dir, 'params'),
                model_file = path.join(phy_model_dir, 'model' .. max_iter),
            },
            tmp_dir = {
                dir = phy_tmp_dir,
                decoder_output = path.join(phy_tmp_dir, 'decode.txt'),
                top_response = path.join(phy_tmp_dir, 'top_response.txt'),
            },
            distill_dir = {
                dir = phy_distill_dir,
                train_file = path.join(phy_distill_dir, train_file),
                dev_file = path.join(phy_distill_dir, dev_file),
                test_file = path.join(phy_distill_dir, test_file),
            },
        }
    end

    local params = self.params
    local dirs = {}

    for round = 1, params.rounds do
        local round_dir = template(round, params, self.atten_params)
        if round > 1 then
            local prev = dirs[round - 1]
            round_dir.data_dir = prev.distill_dir
        else
            round_dir.data_dir = {
                train_file = params.train_file,
                dev_file = params.dev_file,
                test_file = params.test_file,
            }
        end
        table.insert(dirs, round_dir)
    end
    return dirs
end

function DistillModelPool:find_dict_file()
    local candidates = {}
    local params_to_look_for = {
        'params', -- Top level pool params.
        'atten_params',
        'decoder_params',
        'distiller_params',
    }

    for _, params in ipairs(params_to_look_for) do
        local dp = self[params].dictPath
        if dp ~= nil then
            table.insert(candidates, { dictPath = dp, params = params })
        end
    end

    assert(#candidates > 0, 'impossible! no dictPath!')

    local function normalize(p)
        return path.normpath(path.abspath(p))
    end

    local the_one = normalize(candidates[1])
    for _, cand in ipairs(candidates) do
        local dp = normalize(cand.dictPath)
        if dp ~= the_one then
            logger.warn('overriding dictPath %s from %s', dp, cand.params)
        end
    end
    return the_one
end

function DistillModelPool:__init(params)
    self.params = params
    self.atten_params = self:load_atten_params(params)
    self.decoder_params = self:load_decoder_params(params)
    self.distiller_params = self:load_distiller_params(params)

    self.dirs = self:create_symbolic_dirs()
    self.dict_file = params.dictPath
end

function DistillModelPool:train()
    for round = 1, self.params.rounds do
        logger.info('Round %d:', round)
        self.round_dir = self.dirs[round]
        logger.info(self.round_dir)

        logger.info('Create directories')
        self:create_physical_dirs()

        logger.info('Train attention model')
        self:train_atten()

        logger.info('Decode')
        self:decode()

        logger.info('Extract the most common responses')
        self:extract_top()

        logger.info('Distill the data')
        self:distill()
    end
end
