require "cutorch"
require "nn"
require "cunn"
require "nngraph"
require 'logroll'

cutorch.manualSeed(123)

local tds = require('tds')
local Dataset = require("Atten/Dataset")

local AttenModel = torch.class('AttenModel')
local logger = logroll.print_logger()


function AttenModel:__init(params)
    self.dataset = Dataset.new(params)
    self.params = params

    self.lstm_source = self:lstm_source_()
    self.lstm_target = self:lstm_target_()

    if self.mode ~= "test" then
        self.lstms_s = self:g_cloneManyTimes(self.lstm_source, self.params.source_max_length)
        self.lstms_t = self:g_cloneManyTimes(self.lstm_target, self.params.target_max_length)
    else
        self.lstms_s = self:g_cloneManyTimes(self.lstm_source, 1)
        self.lstms_t = self:g_cloneManyTimes(self.lstm_target, 1)
    end

    self.softmax = self:softmax_()
    self.Modules = {}
    self.Modules[#self.Modules + 1] = self.lstm_source
    self.Modules[#self.Modules + 1] = self.lstm_target
    self.Modules[#self.Modules + 1] = self.softmax
    self.store_s = {}
    self.store_t = {}

    if self.params.dictPath ~= "" and self.params.dictPath ~= nil then
        self:ReadDict()
    end
end

-- Read a dictionary file in the standard format, store index, word pair in self.dict.
function AttenModel:ReadDict()
    local filename = self.params.dictPath
    logger.info('loading dictionary from %s', filename)

    self.dict = tds.hash()
    local open = assert(io.open(filename, "r"), string.format('cannot open file %s', filename))
    local index = 0

    while true do
        index = index + 1
        local line = open:read("*line")
        if line == nil then break end
        self.dict[index] = line
    end
    logger.info('Done.')
end

function AttenModel:PrintMatrix(matrix)
    for i = 1, matrix:size(1) do
        print(self:IndexToWord(matrix[i]))
    end
    print("\n")
end

-- Given a vector of ids, turn it into a string of comma-separated words.
-- OOV words are presented as a number, the id. If a 2D matrix passed in, it is flatten first.
-- Example:
-- [1, 2, 3] => "a cat sat"
-- [[1, 2, 3]] => "a cat sat"
function AttenModel:IndexToWord(vector)
    if vector:nDimension() == 2 then
        vector = torch.reshape(vector, vector:size(2))
    end
    local string = ""
    for i = 1, vector:size(1) do
        if self.dict[vector[i]] ~= nil then
            -- In vocab
            string = string .. self.dict[vector[i]] .. " "
        else
            -- Out of vocab
            string = string .. vector[i] .. " "
        end
    end
    string = stringx.strip(string)
    return string
end

-- Make a copy of a Tensor up to 3D.
function AttenModel:copy(A)
    local B
    if A:nDimension() == 1 then
        B = torch.Tensor(A:size(1)):cuda()
    end
    if A:nDimension() == 2 then
        B = torch.Tensor(A:size(1), A:size(2)):cuda()
    end
    if A:nDimension() == 3 then
        B = torch.Tensor(A:size(1), A:size(2), A:size(3)):cuda()
    end
    B:copy(A)
    return B
end

function AttenModel:clone_(A)
    local B = {}
    for i = 1, #A do
        if A[i]:nDimension() == 2 then
            B[i] = torch.Tensor(A[i]:size(1), A[i]:size(2)):cuda()
        else
            B[i] = torch.Tensor(A[i]:size(1)):cuda()
        end
        B[i]:copy(A[i])
    end
    return B
end

function AttenModel:g_cloneManyTimes(net, T)
    local clones = {}
    for t = 1, T do
        clones[t] = net:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    return clones
end

function AttenModel:attention()
    local inputs = {}
    local target_t = nn.Identity()()
    local context = nn.Identity()()
    local context_mask = nn.Identity()()

    table.insert(inputs, target_t)
    table.insert(inputs, context)
    table.insert(inputs, context_mask)

    local context_mask_p = nn.MulConstant(1e8)(nn.AddConstant(-1)(context_mask))
    local atten = nn.MM()({ context, nn.Replicate(1, 3)(target_t) })
    atten = nn.Sum(3)(atten)
    atten = nn.CAddTable() { atten, context_mask_p }
    atten = nn.SoftMax()(atten)
    atten = nn.Replicate(1, 2)(atten)

    local context_combined = nn.MM()({ atten, context })
    context_combined = nn.Sum(2)(context_combined)

    local output1 = nn.Linear(self.params.dimension, self.params.dimension, false)(context_combined)
    local output2 = nn.Linear(self.params.dimension, self.params.dimension, false)(inputs[1])
    local output = nn.Tanh()(nn.CAddTable()({ output1, output2 }))
    return nn.gModule(inputs, { output })
end

-- table.insert(table, pos, value) inserts value before pos.
-- table.insert(table, value) append value to the end of table.

function AttenModel:lstm_target_()
    local inputs = {}
    for ll = 1, self.params.layers do
        local h_ll = nn.Identity()()
        table.insert(inputs, h_ll)
        local c_ll = nn.Identity()()
        table.insert(inputs, c_ll)
    end
    -- now inputs is [h_ll, c_ll]

    local context, source_mask
    context = nn.Identity()()
    table.insert(inputs, context)
    local x_ = nn.Identity()()
    table.insert(inputs, x_)
    source_mask = nn.Identity()()
    table.insert(inputs, source_mask)
    -- now inputs is [h_ll, c_ll, context, x_, source_mark]

    local outputs = {}
    -- local LookupTable, input_word_embedding
    for ll = 1, self.params.layers do
        local prev_h = inputs[ll * 2 - 1]
        local prev_c = inputs[ll * 2]

        local x
        if ll == 1 then
            x = nn.LookupTable(self.params.vocab_target, self.params.dimension)(x_)
        else
            x = outputs[(ll - 1) * 2 - 1]
        end

        local drop_x = nn.Dropout(self.params.dropout)(x)
        local drop_h = nn.Dropout(self.params.dropout)(inputs[ll * 2 - 1])
        local i2h = nn.Linear(self.params.dimension, 4 * self.params.dimension, false)(drop_x)
        local h2h = nn.Linear(self.params.dimension, 4 * self.params.dimension, false)(drop_h)
        local gates

        if ll == 1 then
            local atten_feed = self:attention()
            atten_feed.name = 'atten_feed'
            local context1 = atten_feed({ inputs[self.params.layers * 2 - 1], context, source_mask })
            local drop_f = nn.Dropout(self.params.dropout)(context1)
            local f2h = nn.Linear(self.params.dimension, 4 * self.params.dimension, false)(drop_f)
            gates = nn.CAddTable()({ nn.CAddTable()({ i2h, h2h }), f2h })
        else
            gates = nn.CAddTable()({ i2h, h2h })
        end

        local reshaped_gates = nn.Reshape(4, self.params.dimension)(gates)
        local sliced_gates = nn.SplitTable(2)(reshaped_gates)
        local in_gate = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
        local in_transform = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
        local forget_gate = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
        local out_gate = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))
        local l1 = nn.CMulTable()({ forget_gate, inputs[ll * 2] })
        local l2 = nn.CMulTable()({ in_gate, in_transform })
        local next_c = nn.CAddTable()({ l1, l2 })
        local next_h = nn.CMulTable()({ out_gate, nn.Tanh()(next_c) })

        table.insert(outputs, next_h)
        table.insert(outputs, next_c)
    end

    local soft_atten = self:attention()
    soft_atten.name = 'soft_atten'
    local soft_vector = soft_atten({ outputs[self.params.layers * 2 - 1], context, source_mask })
    table.insert(outputs, soft_vector)

    local module = nn.gModule(inputs, outputs)
    module:getParameters():uniform(-self.params.init_weight, self.params.init_weight)
    return module:cuda()
end

function AttenModel:SentencePpl()
    self.mode = "test"
    local score = torch.Tensor(self.Word_s:size(1)):fill(0):cuda()
    local num = torch.Tensor(self.Word_s:size(1)):fill(0):cuda()

    for t = self.Word_t:size(2) - 1, 1, -1 do
        local current_word = self.Word_t:select(2, t + 1)
        local softmax_output = self.softmax:forward({ self.softmax_h[t], current_word })
        local err = softmax_output[1]
        for i = 1, self.Word_t:size(1) do
            if self.Padding_t[i][t + 1] == 1 then
                num[i] = num[i] + 1
                score[i] = score[i] + softmax_output[2][i][current_word[i]]
            end
        end
    end

    for i = 1, self.Word_s:size(1) do
        score[i] = score[i] / num[i]
    end
    return score
end

function AttenModel:softmax_()
    local y = nn.Identity()()
    local h = nn.Identity()()
    local h2y = nn.Linear(self.params.dimension, self.params.vocab_target):noBias()(h)
    local pred = nn.LogSoftMax()(h2y)
    local w = torch.ones(self.params.vocab_target)

    w[self.dataset.target_dummy] = 0
    local Criterion = nn.ClassNLLCriterion(w)
    Criterion.sizeAverage = false
    local err = Criterion({ pred, y })
    local module = nn.gModule({ h, y }, { err, pred })

    module:getParameters():uniform(-self.params.init_weight, self.params.init_weight)
    return module:cuda()
end

function AttenModel:lstm_source_()
    local inputs = {}
    for ll = 1, self.params.layers do
        table.insert(inputs, nn.Identity()())
        table.insert(inputs, nn.Identity()())
    end

    table.insert(inputs, nn.Identity()())
    local outputs = {}

    for ll = 1, self.params.layers do
        local prev_h = inputs[ll * 2 - 1]
        local prev_c = inputs[ll * 2]
        local x
        if ll == 1 then
            x = nn.LookupTable(self.params.vocab_source, self.params.dimension)(inputs[#inputs])
        else
            x = outputs[(ll - 1) * 2 - 1]
        end

        local drop_x = nn.Dropout(self.params.dropout)(x)
        local drop_h = nn.Dropout(self.params.dropout)(inputs[ll * 2 - 1])
        local i2h = nn.Linear(self.params.dimension, 4 * self.params.dimension, false)(drop_x)
        local h2h = nn.Linear(self.params.dimension, 4 * self.params.dimension, false)(drop_h)
        local gates = nn.CAddTable()({ i2h, h2h })
        local reshaped_gates = nn.Reshape(4, self.params.dimension)(gates)
        local sliced_gates = nn.SplitTable(2)(reshaped_gates)
        local in_gate = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
        local in_transform = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
        local forget_gate = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
        local out_gate = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))
        local l1 = nn.CMulTable()({ forget_gate, inputs[ll * 2] })
        local l2 = nn.CMulTable()({ in_gate, in_transform })
        local next_c = nn.CAddTable()({ l1, l2 })
        local next_h = nn.CMulTable()({ out_gate, nn.Tanh()(next_c) })

        table.insert(outputs, next_h)
        table.insert(outputs, next_c)
    end

    local module = nn.gModule(inputs, outputs)
    module:getParameters():uniform(-self.params.init_weight, self.params.init_weight)
    return module:cuda()
end

function AttenModel:model_forward()
    self.context = torch.Tensor(self.Word_s:size(1), self.Word_s:size(2), self.params.dimension):cuda()
    self.target_embedding = {}
    self.softmax_h = {}
    local output

    local timer = torch.Timer()

    for t = 1, self.Word_s:size(2) do
        local input = {}
        if t == 1 then
            for ll = 1, self.params.layers do
                table.insert(input, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
                table.insert(input, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
            end
        else
            if self.mode == "train" then
                input = self:clone_(self.store_s[t - 1])
            else
                input = self:clone_(output)
            end
        end

        table.insert(input, self.Word_s:select(2, t))
        if self.mode == "train" then
            self.lstms_s[t]:training()
            output = self.lstms_s[t]:forward(input)
        else self.lstms_s[1]:evaluate()
            output = self.lstms_s[1]:forward(input)
        end

        if self.Mask_s[t]:nDimension() ~= 0 then
            for i = 1, #output do
                output[i]:indexCopy(1, self.Mask_s[t],
                    torch.zeros(self.Mask_s[t]:size(1), self.params.dimension):cuda())
            end
        end

        if self.mode == "train" then
            self.store_s[t] = self:clone_(output)
        elseif t == self.Word_s:size(2) then
            self.last = output
        end

        self.SourceVector = output[self.params.layers * 2 - 1]
        self.context[{ {}, t }]:copy(output[2 * self.params.layers - 1])
    end

    if self.mode ~= "decoding" then
        for t = 1, self.Word_t:size(2) - 1 do
            local lstm_input = {}
            if t == 1 then
                if self.mode == "train" then
                    lstm_input = self:clone_(self.store_s[self.Word_s:size(2)])
                else
                    lstm_input = output
                end
            else
                if self.mode == "train" then
                    lstm_input = self:clone_(self.store_t[t - 1])
                else
                    lstm_input = {}
                    for i = 1, 2 * self.params.layers do
                        lstm_input[i] = output[i]
                    end
                end
            end

            table.insert(lstm_input, self.context)
            table.insert(lstm_input, self.Word_t:select(2, t))
            table.insert(lstm_input, self.Padding_s)

            -- For Persona, absent key result in nil in Lua.
            if self.params.speakerSetting == "speaker" or
                    self.params.speakerSetting == "speaker_addressee" then
                table.insert(lstm_input, self.SpeakerID)
            end
            if self.params.speakerSetting == "speaker_addressee" then
                table.insert(lstm_input, self.AddresseeID)
            end

            if self.mode == "train" then
                self.lstms_t[t]:training()
                output = self.lstms_t[t]:forward(lstm_input)
            else
                self.lstms_t[1]:evaluate()
                logger.debug(lstm_input)
                output = self.lstms_t[1]:forward(lstm_input)
            end

            if self.Mask_t[t]:nDimension() ~= 0 then
                for i = 1, #output do
                    output[i]:indexCopy(1, self.Mask_t[t],
                        torch.zeros(self.Mask_t[t]:size(1), self.params.dimension):cuda())
                end
            end

            self.store_t[t] = {}
            for i = 1, #output - 1 do
                self.store_t[t][i] = self:copy(output[i])
            end

            self.softmax_h[t] = self:copy(output[#output])
        end
    end
end

function AttenModel:model_backward()
    local d_source = torch.zeros(self.context:size(1), self.context:size(2), self.context:size(3)):cuda()
    local d_output = {}

    for ll = 1, self.params.layers do
        table.insert(d_output, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
        table.insert(d_output, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
    end

    local sum_err = 0
    local total_num = 0

    for t = self.Word_t:size(2) - 1, 1, -1 do
        local current_word = self.Word_t:select(2, t + 1)
        local softmax_output = self.softmax:forward({ self.softmax_h[t], current_word })
        local err = softmax_output[1]

        sum_err = sum_err + err[1]
        total_num = total_num + self.Left_t[t + 1]:size(1)

        if self.mode == "train" then
            local dh = self.softmax:backward({ self.softmax_h[t], current_word },
                { torch.Tensor({ 1 }), torch.Tensor(softmax_output[2]:size()):fill(0):cuda() })

            local d_store_t = self:clone_(d_output)
            table.insert(d_store_t, dh[1])

            local now_input = {}
            if t ~= 1 then
                now_input = self:clone_(self.store_t[t - 1])
            else
                now_input = self:clone_(self.store_s[self.Word_s:size(2)])
            end

            table.insert(now_input, self.context)
            table.insert(now_input, self.Word_t:select(2, t))
            table.insert(now_input, self.Padding_s)

            local now_d_input = self.lstms_t[t]:backward(now_input, d_store_t)
            if self.Mask_t[t]:nDimension() ~= 0 then
                for i = 1, 2 * self.params.layers + 2 do
                    now_d_input[i]:indexCopy(1, self.Mask_t[t],
                        torch.zeros(self.Mask_t[t]:size(1), self.params.dimension):cuda())
                end
            end

            d_output = {}
            for i = 1, 2 * self.params.layers do
                d_output[i] = self:copy(now_d_input[i])
            end
            d_source:add(now_d_input[2 * self.params.layers + 1])
        end
    end

    if self.mode == "train" then
        for t = self.Word_s:size(2), 1, -1 do
            local now_input = {}
            if t ~= 1 then
                now_input = self:clone_(self.store_s[t - 1])
            else
                for ll = 1, self.params.layers do
                    table.insert(now_input, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
                    table.insert(now_input, torch.zeros(self.Word_s:size(1), self.params.dimension):cuda())
                end
            end

            table.insert(now_input, self.Word_s:select(2, t))
            d_output[2 * self.params.layers - 1]:add(d_source[{ {}, t, {} }])
            local d_now_output = self.lstms_s[t]:backward(now_input, d_output)

            if self.Mask_s[t]:nDimension() ~= 0 then
                for i = 1, #d_now_output - 1 do
                    d_now_output[i]:indexCopy(1,
                        self.Mask_s[t], torch.zeros(self.Mask_s[t]:size(1), self.params.dimension):cuda())
                end
            end

            d_output = {}
            for i = 1, 2 * self.params.layers do
                d_output[i] = self:copy(d_now_output[i])
            end
        end
    end
    return sum_err, total_num
end

function AttenModel:update()
    local lr
    if self.lr ~= nil then
        lr = self.lr
    else
        lr = self.params.alpha
    end

    local grad_norm = 0
    for i = 1, #self.Modules do
        local p, dp = self.Modules[i]:parameters()
        for j, m in pairs(dp) do
            m:mul(1 / self.Word_s:size(1))
            grad_norm = grad_norm + m:norm() ^ 2
        end
    end

    grad_norm = grad_norm ^ 0.5

    if grad_norm > self.params.thres then
        lr = lr * self.params.thres / grad_norm
    end

    for i = 1, #self.Modules do
        self.Modules[i]:updateParameters(lr)
    end
end

function AttenModel:save()
    local weights = {}
    for i = 1, #self.Modules do
        weights[i] = self.Modules[i]:parameters()
    end

    local filename = string.format('%s%d', self.params.save_prefix, self.iter)
    logger.info('Saving module weights to %s', filename)

    local file = torch.DiskFile(filename, "w"):binary()
    file:writeObject(weights)
    file:close()
end

function AttenModel:saveParams()
    logger.info(self.params)
    local filename = self.params.save_params_file
    logger.info('Saving model hyper parameters to %s', filename)
    local file = torch.DiskFile(filename, "w"):binary()
    file:writeObject(self.params)
    file:close()
end

function AttenModel:readModel()
    logger.info('loading model weights from %s', self.params.model_file)
    local file = torch.DiskFile(self.params.model_file, "r"):binary()
    local model_params = file:readObject()
    file:close()

    for i = 1, #self.Modules do
        local parameter, _ = self.Modules[i]:parameters()
        for j = 1, #parameter do
            parameter[j]:copy(model_params[i][j])
        end
    end
    logger.info("read model done")
end

function AttenModel:clear()
    for i = 1, #self.Modules do
        self.Modules[i]:zeroGradParameters()
    end
end


function AttenModel:test()
    local open_train_file
    if self.mode == "dev" then
        logger.info('Using develop file')
        open_train_file = assert(io.open(self.params.dev_file, "r"), 'cannot open file')
    elseif self.mode == "test" then
        logger.info('Using test file')
        open_train_file = assert(io.open(self.params.test_file, "r"), 'cannot open file')
    end

    local sum_err_all = 0
    local total_num_all = 0
    local End = 0
    while End == 0 do
        End, self.Word_s, self.Word_t,
        self.Mask_s, self.Mask_t,
        self.Left_s, self.Left_t,
        self.Padding_s, self.Padding_t = self.dataset:read_train(open_train_file)

        if #self.Word_s == 0 or End == 1 then
            break
        end

        if (self.Word_s:size(2) < self.params.source_max_length and
                self.Word_t:size(2) < self.params.target_max_length) then
            self.mode = "test"
            self.Word_s = self.Word_s:cuda()
            self.Word_t = self.Word_t:cuda()
            self.Padding_s = self.Padding_s:cuda()
            self:model_forward()
            local sum_err, total_num = self:model_backward()
            sum_err_all = sum_err_all + sum_err
            total_num_all = total_num_all + total_num
        end
    end

    open_train_file:close()
    local ppl = 1 / torch.exp(-sum_err_all / total_num_all)
    logger.info('standard perplexity: %f', ppl)
end

function AttenModel:train()
    if self.params.saveModel then
        logger.info('Saving hyper parameters...')
        self:saveParams()
    end

    local timer = torch.Timer()
    self.iter = 0
    local start_halving = false
    self.lr = self.params.alpha

    logger.info('Initial testing...')
    self.mode = "test"
    self:test()

    while true do
        logger.info("Epoch: %d", self.iter)
        self.iter = self.iter + 1

        if self.params.start_halve ~= -1 then
            if self.iter > self.params.start_halve then
                start_halving = true
            end
        end
        if start_halving then
            self.lr = self.lr * 0.5
        end

        local open_train_file = assert(io.open(self.params.train_file, "r"), 'cannot open file')
        local End, Word_s, Word_t, Mask_s, Mask_t
        local End = 0
        local batch_n = 1
        local time1 = timer:time().real

        while End == 0 do
            batch_n = batch_n + 1
            self:clear()
            logger.info('loading training dataset %s', open_train_file)
            End, self.Word_s, self.Word_t, self.Mask_s, self.Mask_t,
            self.Left_s, self.Left_t, self.Padding_s, self.Padding_t = self.dataset:read_train(open_train_file)
            if End == 1 then
                break
            end

            local train_this_batch = false
            if (self.Word_s:size(2) < 60 and self.Word_t:size(2) < 60) then
                train_this_batch = true
            end

            if train_this_batch then
                self.mode = "train"
                local time1 = timer:time().real
                self.Word_s = self.Word_s:cuda()
                self.Word_t = self.Word_t:cuda()
                self.Padding_s = self.Padding_s:cuda()

                logger.info('Forward pass')
                self:model_forward()
                logger.info('Backward pass')
                self:model_backward()
                logger.info('Update pass')
                self:update()
                local time2 = timer:time().real
            end
        end

        open_train_file:close()
        logger.info('Running validation test...')
        self.mode = "test"
        self:test()
        if self.params.saveModel then
            self:save()
        end

        local time2 = timer:time().real
        logger.info("Batch Time: %f", time2 - time1)

        if self.iter == self.params.max_iter then
            logger.info("Done training!")
            break
        end
    end
end

return AttenModel
