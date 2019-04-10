require "cutorch"
require "nn"
require 'cunn'
require "nngraph"
require 'logroll'

local logger = logroll.print_logger()
local GenModel = require('Adversarial/Reinforce/GenModel')
local DisModel = require('Adversarial/Discriminative/Model')
local RLModel = torch.class('RLModel')

function RLModel:LoadGenModel()
    local filename = self.params.generate_params
    local file = torch.DiskFile(filename, "r"):binary()
    logger.info('loading generative_params from %s', filename)
    local generate_params = file:readObject()
    file:close()

    generate_params.sample = self.params.sample
    generate_params.alpha = generate_params.alpha * self.params.lr
    generate_params.model_file = self.params.generate_model
    generate_params.batch_size = self.params.batch_size
    generate_params.vanillaReinforce = self.params.vanillaReinforce
    generate_params.MonteCarloExample_N = self.params.MonteCarloExample_N
    generate_params.max_length = 40
    generate_params.min_length = 0
    assert(generate_params.dictPath and path.isfile(generate_params.dictPath))
    generate_params.output_source_target_side_by_side = true
    generate_params.PrintOutIllustrationSample = true
    generate_params.beam_size = 1
    generate_params.target_length = 0
    generate_params.NBest = false
    generate_params.train_file = self.params.trainData
    generate_params.dev_file = self.params.devData
    generate_params.test_file = self.params.testData

    -- For GenModel to know where to save things.
    generate_params.saveFolder = path.join(self.params.saveFolder, 'gen')
    generate_params.save_prefix = path.join(generate_params.saveFolder, 'model')
    generate_params.save_params_file = path.join(generate_params.saveFolder, 'params')

    if not path.isdir(generate_params.saveFolder) then
        logger.info('mkdir %s', generate_params.saveFolder)
        paths.mkdir(generate_params.saveFolder)
    end

    logger.info("generative_params:")
    logger.info(generate_params)

    logger.info('creating GenModel from generative_params...')
    self.generate_model = GenModel.new(generate_params)

    logger.info('loading weights for GenModel...')
    self.generate_model:readModel()
end

function RLModel:LoadDisModel()
    local filename = self.params.disc_params
    local file = torch.DiskFile(filename, "r"):binary()
    logger.info('loading disc_params from %s', filename)
    local disc_params = file:readObject()
    file:close()

    -- Override --
    disc_params.model_file = self.params.disc_model
    disc_params.source_max_length = 100
    disc_params.batch_size = self.params.batch_size
    disc_params.trainData = self.params.trainData
    disc_params.devData = self.params.devData
    disc_params.testData = self.params.testData

    -- For DisModel to know where to save things.
    disc_params.saveFolder = path.join(self.params.saveFolder, 'dis')

    logger.info('disc_params:')
    logger.info(disc_params)

    logger.info('creating DisModel from disc_params...')
    self.disc_model = DisModel.new(disc_params)

    logger.info('loading weights for DisModel...')
    self.disc_model:readModel()
    self.disc_model.mode = "train"
end

function RLModel:__init(params)
    params.save_params_file = path.join(params.saveFolder, 'params')
    params.output_file = path.join(params.saveFolder, 'log')

    if not path.isdir(params.saveFolder) then
        logger.info('mkdir %s', params.saveFolder)
        paths.mkdir(params.saveFolder)
    end
    self.params = params

    self:LoadGenModel()
    self:LoadDisModel()

    self.base_line_sum = 0
    self.base_line_n = 0
    self.log_ = assert(io.open(self.params.output_file, "w"), 'cannot open output_file')

    if self.params.baseline and self.params.baselineType == "critic" then
        logger.info('creating baseline model...')
        self.baseline = nn.Linear(self.params.dimension, 1)
        self.baseline.weight:fill(0)
        self.baseline.bias:fill(0)
        self.baseline:cuda()
        self.baseline_mse = nn.MSECriterion()
        self.baseline_mse:cuda()
    end
end


function RLModel:save()
    -- save both GenModel and DisModel.
    local models_to_save = { self.generate_model, self.disc_model }

    for i, model in ipairs(models_to_save) do
        logger.info('Saving weights for %s', model)
        model:save(self.iter)
        logger.info('Saving params for %s', model)
        model:saveParams()
    end

    -- save params of the RLModel.
    local filename = self.params.save_params_file
    local file = torch.DiskFile(filename, "w"):binary()
    logger.info('Saving params for %s to %s', self, filename)
    file:writeObject(self.params)
    file:close()
end

function RLModel:decodeSample()
    logger.info('Using sampling')
    self.generate_model.params.setting = "sampling"
    self.generate_model:DecodeIllustrationSample(self.log_)

    logger.info('Using beam search')
    self.generate_model.params.setting = "BS"
    self.generate_model:DecodeIllustrationSample(self.log_)
    self.generate_model.params.setting = "sampling"
end

function RLModel:trainD(train_file)
    local End = 0
    self.generate_model:clear()

    End, self.generate_model.Word_s, self.generate_model.Word_t,
    self.generate_model.Mask_s, self.generate_model.Mask_t,
    self.generate_model.Left_s, self.generate_model.Left_t,
    self.generate_model.Padding_s, self.generate_model.Padding_t,
    self.generate_model.Source, self.generate_model.Target = self.generate_model.dataset:read_train(train_file)

    if End == 1 then
        return End
    end

    self.generate_model.mode = "decoding"
    self.generate_model:GenerateSample()

    local Source = {}
    Source[1] = {}
    Source[2] = {}

    self.disc_model.labels = torch.Tensor(#self.generate_model.Source * 2):cuda()
    self.disc_model.labels:sub(1, #self.generate_model.Source):fill(1)
    self.disc_model.labels:sub(1 + #self.generate_model.Source, 2 * #self.generate_model.Source):fill(2)
    self.disc_model.labels = self.disc_model.labels:cuda()

    for i = 1, #self.generate_model.Source do
        Source[1][#Source[1] + 1] = self.generate_model.Source[i]
        Source[2][#Source[2] + 1] = self.generate_model.Target[i]:sub(1, 1, 2, self.generate_model.Target[i]:size(2) - 1)
    end

    for i = 1, #self.generate_model.Source do
        Source[1][#Source[1] + 1] = self.generate_model.Source[i]
        Source[2][#Source[2] + 1] = self.generate_model.sample_target[i]:sub(1, 1, 2, self.generate_model.sample_target[i]:size(2) - 1)
    end

    self.disc_model.Word_s, self.disc_model.Mask_s,
    self.disc_model.Left_t, self.disc_model.Padding_s = self.disc_model.dataset:get_batch(Source)

    for i, v in pairs(self.disc_model.Word_s) do
        v = v:cuda()
    end

    self.disc_model:clear()
    self.disc_model:model_forward()
    local dis_pred = self.disc_model:model_backward()

    if self.disc_model.mode == "train" then
        self.disc_model:update()
    end

    return End, dis_pred
end

function RLModel:train()
    self.iter = 0

    while true do
        logger.info('Epoch: %d', self.iter)
        self.iter = self.iter + 1

        local train_file = assert(io.open(self.params.trainData, "r"),
            'cannot open trainData')
        local End = 0

        if self.iter % self.params.logFreq == 0 then
            self.log_:write(string.format('Epoch: %d\n', self.iter))
            logger.info('[Validate] decoding samples...')
            self:decodeSample()

            logger.info('[Validate] testing...')
            self.generate_model.mode = "test"
            self.generate_model:test()

            logger.info('[Validate] saving models...')
            self:save()
        end

        local batch_n = 0
        while End == 0 do
            batch_n = batch_n + 1
            logger.info('train DisModel...')
            for i = 1, self.params.dSteps do
                logger.info('D-epoch: %d', i)
                self.disc_model.mode = "train"
                End = self:trainD(train_file)
                if End == 1 then
                    break
                end
            end

            if End == 0 then
                logger.info('train GenModel...')
                for i = 1, self.params.gSteps do
                    logger.info('G-epoch: %d', i)
                    self.disc_model.mode = "test"
                    local End, dis_pred = self:trainD(train_file)
                    if End == 1 then
                        break
                    end

                    dis_pred = torch.exp(dis_pred)
                    local reward = dis_pred:sub(1 + self.generate_model.params.batch_size,
                        2 * self.generate_model.params.batch_size, 1, 1)

                    logger.info('Integrate...')
                    self.generate_model:Integrate(true)
                    if not self.params.vanillaReinforce then
                        self:MonteCarloReward()
                    end

                    logger.info('Reinforce...')
                    self:Reinforce(reward)
                end

                if self.params.TeacherForce then
                    logger.info('Teacher forcing...')
                    self.generate_model:Integrate(false)
                    self.generate_model:clear()
                    self.generate_model.mode = "train"
                    self.generate_model:model_forward()
                    self.generate_model:model_backward(true)
                    self.generate_model:update()
                end
            end
        end

        train_file:close()

        if self.params.saveModel then
            self:save()
        end

        if self.iter == self.params.max_iter then
            break
        end
    end
end

function RLModel:Reinforce(reward)
    reward = torch.reshape(reward, reward:size(1))
    self.base_line_sum = self.base_line_sum + reward:sum()
    self.base_line_n = self.base_line_n + reward:size(1)
    local baseline

    if self.params.baseline and self.params.baselineType == "critic" then
        local baseline_input = self.generate_model.SourceVector
        baseline = self.baseline:forward(baseline_input)
        local baseline_error = self.baseline_mse:forward(baseline, reward)
        self.baseline:zeroGradParameters()
        self.baseline_mse:backward(baseline, reward)
        self.baseline:backward(baseline_input, self.baseline_mse.gradInput)
        self.baseline:updateParameters(self.params.baseline_lr)
    elseif self.params.baseline and self.params.baselineType == "aver" then
        baseline = self.base_line_sum / self.base_line_n
    end

    if self.params.baseline then
        self.generate_model.last_reward = self.params.Timeslr * (-reward + baseline) / self.params.batch_size
        if not self.params.vanillaReinforce then
            baseline = torch.repeatTensor(baseline, 1, self.generate_model.MonteCarloReward:size(2))
            self.generate_model.MonteCarloReward = self.params.Timeslr * (-self.generate_model.MonteCarloReward + baseline) / self.params.batch_size
        end
    else
        self.generate_model.last_reward = -self.params.Timeslr * reward / self.params.batch_size
        if not self.params.vanillaReinforce then
            self.generate_model.MonteCarloReward = -self.params.Timeslr * self.generate_model.MonteCarloReward / self.params.batch_size
        end
    end

    self.generate_model.mode = "train"
    self.generate_model:model_forward()
    self.generate_model:model_backward(false)
    self.generate_model:update(false)
end

function RLModel:MonteCarloReward()
    local Source = {}
    Source[1] = {}
    Source[2] = {}

    for j = 1, #self.generate_model.Source do
        for i = 1, self.params.MonteCarloExample_N do
            Source[1][#Source[1] + 1] = self.generate_model.Source[j]
        end
    end

    for t = 1, self.generate_model.Word_t:size(2) - 2 do
        self.generate_model:ComputeMonteCarloReward(t)
        Source[2] = {}

        for j = 1, #self.generate_model.partial_history do
            Source[2][#Source[2] + 1] = self.generate_model.partial_history[j]
        end

        self.disc_model.Word_s, self.disc_model.Mask_s,
        self.disc_model.Left_t, self.disc_model.Padding_s = self.disc_model.dataset:get_batch(Source)

        for i, v in pairs(self.disc_model.Word_s) do
            v = v:cuda()
        end

        self.disc_model.mode = "test"
        self.disc_model:model_forward()
        self.disc_model.labels = torch.Tensor(self.disc_model.Word_s[1]:size(1)):fill(1):cuda()

        local dis_pred = self.disc_model:model_backward()
        dis_pred = torch.exp(dis_pred)

        local reward_monte_carlo = dis_pred:select(2, 1)
        reward_monte_carlo = torch.reshape(reward_monte_carlo, #self.generate_model.Source, self.params.MonteCarloExample_N)
        reward_monte_carlo = reward_monte_carlo:mean(2)

        if t == 1 then
            self.generate_model.MonteCarloReward = reward_monte_carlo
        else
            self.generate_model.MonteCarloReward = torch.cat(self.generate_model.MonteCarloReward, reward_monte_carlo, 2)
        end
    end
end

return RLModel
