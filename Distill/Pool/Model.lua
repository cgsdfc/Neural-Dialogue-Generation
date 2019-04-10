-- ===================
-- Overview of a Round
-- ===================
-- This module implements a simple system to run multiple rounds of data distillation
-- consecutively. In each round, an attention model is first trained on the data remained
-- after previous round of distillation (or the initial data if it is the first round).
-- Then the model is used to Decode a large number of context
-- to response on the training set. For decoding, which set of data to use is not important and we
-- choose the training set since it is large. Then the most frequent responses of the decoder output are
-- extracted. Denoted as *top_response*, these responses are used as a filter to distill the dataset.
--
-- For the distillation step, you can choose from two distiller -- Glove and Encoder. They differ in
-- the way to convert sentences to embeddings. Glove uses the pre-trained *Global Vector* (Pennington et al. 2018)
-- a variant of word embedding, to map each word to a vector and use the sum of the words as the representation of a
-- sentence. Encoder uses the encoder part of a Seq2Seq model to obtain a vector representation of the sentence.
-- It is also better known as the hidden state of the last time step in the LSTM unit.
-- Each distiller accepts their own options while they both accept some common options like distill_rate. Refer
-- to their document for further information.
--
-- A distiller first assigns a *similarity score* to every examples in the data to distill. The score is based on
-- cosine similarity and in many ways resembles the *embedding-based metric*. Then the examples with the highest
-- score wihin the range allowed by distill_rate will be removed from the original set. All the train, develop and test dataset
-- are distilled in this way and they become the data for the next round. This completes one round of distillation.
--
--
-- ======================
-- Implementation Details
-- ======================
-- The implementation makes use of a *symbolic directory* to run the system.
-- A symbolic directory is a nested table holded in memory and backed by physical directories.
-- The keys of the table represent the symbolic dirs or files while the values of the table represent
-- phisical directories or files.
--
-- This design allows the code interfacing each models
-- to manipulate directory and file in a simple syntax, using dot notation. Since the number of models
-- we interface is more than a few, this greatly simplifies things.
-- It also separates the creation of physical paths and the use of them. We make sure all paths needed in an invocation
-- of a model are created before.
-- This renders clean, readable and shorter code, free of the cluster of mkdir and path.join.
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
--      * distill_dir
--          * train_file
--          * dev_file
--          * test_file
--
-- This structure describes all things required to run the system.
-- An entry might not have a phisical correspondence.
-- All the round_dirs for all rounds are kept in the memory in the whole training.
-- Specially, round-1, which is the initial round, maps its data_dir and files
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

-- Train an attention model.
function DistillModelPool:train_atten()
    local params = self.atten_params
    local round_dir = self.round_dir
    local data_dir = round_dir.data_dir

    for key, value in pairs(data_dir) do
        if key ~= 'dir' then
            params[key] = value
        end
    end
    params.saveFolder = round_dir.model_dir.dir

    logger.info('atten_params:')
    print(params)
    local model = AttenModel.new(params)
    model:train()
end

-- Decode the training set using the attention model.
function DistillModelPool:decode()
    local params = self.decoder_params
    local round_dir = self.round_dir

    params.model_file = round_dir.model_dir.model_file
    params.params_file = round_dir.model_dir.params_file
    params.InputFile = round_dir.data_dir.train_file
    params.OutputFile = round_dir.tmp_dir.decoder_output

    logger.info('decoder_params:')
    print(params)
    local decoder = Decoder.new(params)
    decoder:decode()
end

-- Extract top_response.
function DistillModelPool:extract_top()
    local input = self.round_dir.tmp_dir.decoder_output
    local output = self.round_dir.tmp_dir.top_response
    local dict_file = self.dict_file
    local freq_threshold = self.params.freq_threshold

    logger.info('input: %s', input)
    logger.info('output: %s', output)
    logger.info('freq_threshold: %s', freq_threshold)
    extract_top(input, output, dict_file, freq_threshold)
end

-- Distill the data with top_response.
function DistillModelPool:distill()
    local class = self.distiller_class
    local params = self.distiller_params
    local round_dir = self.round_dir

    -- *Note*: we rely on the fact that the distiller use the *same* filename
    -- as its input files.
    params.saveFolder = round_dir.distill_dir.dir
    params.TopResponseFile = round_dir.tmp_dir.top_response

    for key, data_file in pairs(round_dir.data_dir) do
        if key ~= 'dir' then
            params.TrainingData = data_file
            logger.info('distiller_params:')
            print(params)
            local distiller = class.new(params)
            distiller:Distill()
        end
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
    for _, files in pairs(self.round_dir) do
        if files.dir then -- prevent some missing dir.
            mkdir_if_not_yet(files.dir)
        end
    end
end

-- Create all the symbolic dirs we need in all rounds.
function DistillModelPool:create_symbolic_dirs()
    -- Create a template round_dir table.
    local function template(round, params, atten_params)
        local phy_round_dir = path.join(params.saveFolder, tostring(round))

        local phy_model_dir = path.join(phy_round_dir, 'model')
        local phy_distill_dir = path.join(phy_round_dir, 'distill')
        local phy_tmp_dir = path.join(phy_round_dir, 'tmp')

        local train_file = path.basename(params.train_file)
        local dev_file = path.basename(params.dev_file)
        local test_file = path.basename(params.test_file)

        local max_iter = atten_params.max_iter

        return {
           -- data_dir to be filled later.
            model_dir = {
                dir = phy_model_dir, -- point to itself.
                params_file = path.join(phy_model_dir, 'params'),
                model_file = path.join(phy_model_dir, 'model' .. max_iter),
            },
            tmp_dir = {
                dir = phy_tmp_dir,
                decoder_output = path.join(phy_tmp_dir, 'Decode.txt'),
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
                dir = path.dirname(params.train_file),
                train_file = params.train_file,
                dev_file = params.dev_file,
                test_file = params.test_file,
            }
        end
        table.insert(dirs, round_dir)
    end
    return dirs
end

-- Since many models accept -dictPath option, we need to find out
-- *the_one* we are going to use.
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

    local the_one = normalize(candidates[1].dictPath)
    for _, cand in ipairs(candidates) do
        local dp = normalize(cand.dictPath)
        if dp ~= the_one then -- if there is inconsistency.
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
    self.dict_file = self:find_dict_file()
    logger.info('using dict_file %s', self.dict_file)
end

function DistillModelPool:train()
    local total_round = self.params.rounds
    logger.info('Begin %d rounds of distillation', total_round)

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
    logger.info('All rounds done.')
end

return DistillModelPool
