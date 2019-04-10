require 'Atten/Model'
require 'logroll'

local logger = logroll.print_logger()
local LenModel = torch.class('LenModel', 'AttenModel')
local Dataset = require('Atten/Dataset')


function LenModel:__init(params)
    logger.info('loading params for the pretrained attention model: %s', params.params_file)
    local file = torch.DiskFile(params.params_file, "r"):binary()
    local atten_params = file:readObject()
    file:close()

    for k, v in pairs(atten_params) do
        if params[k] == nil or params[k] == "" then
            params[k] = v
        end
    end
    self.params = params

    logger.info('creating future prediction model')
    self.map = nn.Sequential()
    self.map:add(nn.Linear(self.params.dimension, self.params.dimension))
    self.map:add(nn.Tanh())
    self.map:add(nn.Linear(self.params.dimension, self.params.dimension))
    self.map:add(nn.Tanh())
    self.map:add(nn.Linear(self.params.dimension, 1))
    self.map = self.map:cuda()
    self.mse = nn.MSECriterion()
    self.mse = self.mse:cuda()

    if self.params.readSequenceModel then
        logger.info('reading sequence model')
        self.dataset = Dataset.new(params)

        self.lstm_source = self:lstm_source_()
        self.lstm_target = self:lstm_target_()
        self.softmax = self:softmax_()
        self.Modules = {}
        self.Modules[#self.Modules + 1] = self.lstm_source
        self.Modules[#self.Modules + 1] = self.lstm_target
        self.Modules[#self.Modules + 1] = self.softmax
        self.lstms_s = {}
        self.lstms_t = {}
        self.lstms_s[1] = self.lstm_source
        self.lstms_t[1] = self.lstm_target
        self.store_s = {}

        self:readModel()
    end

    if self.params.readFutureModel then
        logger.info('reading future model')
        local parameter = self.map:parameters()
        local file = torch.DiskFile(self.params.FuturePredictorModelFile, "r"):binary()
        local future_params = file:readObject()
        file:close()

        for j = 1, #parameter do
            parameter[j]:copy(future_params[j])
        end
        logger.info("length predictor intialization done")
    end
end

function LenModel:model_forward()
    local totalError = 0
    local instance = 0
    self.context = torch.Tensor(self.Word_s:size(1), self.Word_s:size(2), self.params.dimension):cuda()
    local output

    -- run the encoder rnn.
    for t = 1, self.Word_s:size(2) do
        local input = {}
        if t == 1 then
            -- Initial time step, zero input.
            for ll = 1, self.params.layers do
                table.insert(input, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
                table.insert(input, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
            end
        else
            -- Feed output as input -- recurrent.
            input = self:clone_(output)
        end

        table.insert(input, self.Word_s:select(2, t))
        self.lstms_s[1]:evaluate()
        output = self.lstms_s[1]:forward(input)

        if self.Mask_s[t]:nDimension() ~= 0 then
            for i = 1, #output do
                output[i]:indexCopy(1, self.Mask_s[t],
                    torch.zeros(self.Mask_s[t]:size(1), self.params.dimension):cuda())
            end
        end

        self.context[{ {}, t }]:copy(output[2 * self.params.layers - 1])
    end

    -- run the decoder rnn.
    for t = 1, self.Word_t:size(2) - 1 do
        local lstm_input
        if t == 1 then
            -- initialize from encoder's last hidden state.
            lstm_input = output
        else
            -- recurrent
            lstm_input = {}
            for i = 1, 2 * self.params.layers do
                lstm_input[i] = output[i]
            end
        end

        table.insert(lstm_input, self.context)
        table.insert(lstm_input, self.Word_t:select(2, t))
        table.insert(lstm_input, self.Padding_s)

        output = self.lstms_t[1]:forward(lstm_input)
        self.Length = self.Length - 1

        local left_index = (torch.range(1, #self.Source)[self.Length:gt(0)]):long()
        local representation_left = output[2 * self.params.layers - 1]:index(1, left_index)
        local length_left = self.Length:index(1, left_index)
        length_left = torch.reshape(length_left, length_left:size(1), 1):cuda()

        -- update the value function
        self.map:zeroGradParameters()
        -- predict len left using lstm output ht.
        local pred = self.map:forward(representation_left)
        local Error = self.mse:forward(pred, length_left)

        -- Backprop.
        if self.mode == "train" then
            self.mse:backward(pred, length_left)
            self.map:backward(representation_left, self.mse.gradInput)
            self.map:updateParameters(self.params.alpha)
        end

        totalError = totalError + Error * length_left:size(1)
        instance = instance + length_left:size(1)
        if self.Mask_t[t]:nDimension() ~= 0 then
            for i = 1, #output do
                output[i]:indexCopy(1, self.Mask_t[t],
                    torch.zeros(self.Mask_t[t]:size(1), self.params.dimension):cuda())
            end
        end
    end

    return totalError, instance
end


function LenModel:test()
    logger.info('test with file %s', self.params.test_file)
    local test_file = assert(io.open(self.params.test_file, "r"), 'cannot open test_file')

    local End, Word_s, Word_t, Mask_s, Mask_t
    local End = 0
    local batch_n = 1
    local totalError = 0
    local n_instance = 0

    while End == 0 do
        batch_n = batch_n + 1
        self:clear()

        End, self.Word_s, self.Word_t,
        self.Mask_s, self.Mask_t,
        self.Left_s, self.Left_t,
        self.Padding_s, self.Padding_t,
        self.Source, self.Target = self.dataset:read_train(test_file)

        if End == 1 then
            break
        end

        self.Length = torch.Tensor(#self.Target):fill(0)
        for i = 1, #self.Target do
            self.Length[i] = self.Target[i]:size(2)
        end

        self.mode = "test"
        self.Word_s = self.Word_s:cuda()
        self.Word_t = self.Word_t:cuda()
        self.Padding_s = self.Padding_s:cuda()

        local batchError, batchInstance = self:model_forward()
        totalError = totalError + batchError
        n_instance = n_instance + batchInstance
    end

    local Error = totalError / n_instance
    logger.info('totalError: %.4f', totalError)
    logger.info('n_instance: %d', n_instance)
    logger.info('Error: %.4f', Error)
end

function LenModel:save(batch_n)
    local params = self.map:parameters()
    local filename = path.join(self.params.save_model_path, 'model' .. batch_n)
    logger.info('saving to %s', filename)

    local file = torch.DiskFile(filename, "w"):binary()
    file:writeObject(params)
    file:close()
end

function LenModel:train()
    local End, Word_s, Word_t, Mask_s, Mask_t
    local End = 0
    local batch_n = 1
    self.iter = 0

    while true do
        End = 0
        logger.info('Epoch: %d', self.iter)
        self.iter = self.iter + 1
        local train_file = assert(io.open(self.params.train_file, "r"), 'cannot open train_file')

        while End == 0 do
            batch_n = batch_n + 1
            if batch_n % 10000 == 0 then
                print(batch_n)
                self:test()
                self:save(batch_n)
            end

            self:clear()

            End, self.Word_s, self.Word_t,
            self.Mask_s, self.Mask_t,
            self.Left_s, self.Left_t,
            self.Padding_s, self.Padding_t,
            self.Source, self.Target = self.dataset:read_train(train_file)

            if End == 1 then
                break
            end

            self.Length = torch.Tensor(#self.Target):fill(0)
            for i = 1, #self.Target do
                self.Length[i] = self.Target[i]:size(2)
            end

            self.mode = "train"
            self.Word_s = self.Word_s:cuda()
            self.Word_t = self.Word_t:cuda()
            self.Padding_s = self.Padding_s:cuda()
            self:model_forward()
        end

        self:test()
        self:save(batch_n)

        if self.iter == self.params.max_iter then
            logger.info("Done training!")
            break
        end
    end
end

return LenModel
