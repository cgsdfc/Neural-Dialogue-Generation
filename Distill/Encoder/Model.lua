require 'logroll'
require 'Decode/Decoder'

local logger = logroll.print_logger()
local EncoderDistiller = torch.class('EncoderDistiller', 'Decoder')

function EncoderDistiller:Distill()
    logger.info('Distillation begins...')
    self:ComputeTopResponse()
    self:ComputeScore()
    self:RemoveExamples()
    logger.info('Distillation done!')
end

function EncoderDistiller:ComputeScore()
    local open_train_file = assert(io.open(self.params.TrainingData, "r"), 'cannot open TrainingData')
    self.score = torch.Tensor():cuda()
    local End = 0
    local num = 0

    -- This loop is similar to that of ComputeTopResponse().
    -- It uses the encoder part of the seq2seq model to generate embeddings for reponses in the
    -- training dataset and the cosine similarity score.
    logger.info('Computing embeddings for training responses and consine similarity score...')
    while End == 0 do
        End, self.Word_s, self.Word_t,
        self.Mask_s, self.Mask_t,
        self.Left_s, self.Left_t,
        self.Padding_s, self.Padding_t = self.dataset:read_train(open_train_file)

        if End == 1 then
            break
        end

        self.mode = "decoding"
        self.Word_s = self.Word_s:cuda()
        self.Padding_s = self.Padding_s:cuda()
        self:model_forward()

        local embed = torch.Tensor(self.last[2 * self.params.layers - 1]:size()):cuda():copy(self.last[2 * self.params.layers - 1])
        embed = nn.Normalize(2):cuda():forward(embed)

        -- Rather than collecting the embedding into a list, it is used immediately to
        -- compare with the embeddings of a top response.

        -- # MM document
        -- MM is *Matric Multiplication*. Why not MatMul?
        -- MM:__init(transA, transB)
        --
        -- function MM:updateOutput(input)
        -- assert(#input == 2, 'input must be a pair of minibatch matrices')

        -- The dot product of two normalized vectors are their cosine similarity.
        -- The max of them denotes the most similar pair.

        -- Note: it computes scores between *one* training embedding and *all* top embeddings, using MM.
        local score = nn.MM(false, true):cuda():forward({ embed, self.TopResponseEmbedding })
        score = torch.max(score, 2)

        -- collect the score into self.score.
        if self.score:nDimension() == 0 then
            self.score = score
        else
            self.score = torch.cat(self.score, score, 1)
        end

        num = num + self.params.batch_size
        if num % 1280000 == 0 then
            logger.info('Processed examples: %d', num)
        end
    end

    logger.info('Determine the examples to remove...')
    -- Flatten self.score to eliminate the B dim.
    self.score = torch.reshape(self.score, self.score:size(1))
    logger.info('Sorting consine similarity scores...')
    local rank_score, index = torch.sort(self.score, true)

    -- remove_indexes[i] == 1 if i is to be removed.
    local remove_indexes = {}
    local num_to_remove = torch.floor(num * self.params.distill_rate)
    logger.info('number of examples to be removed: %d', num_to_remove)

    -- Remove the top 10% not the all.
    for i = 1, num_to_remove do
        remove_indexes[index[i]] = 1
    end
    self.remove_indexes = remove_indexes
end

function EncoderDistiller:RemoveExamples()
    logger.info('Removing examples...')
    local train_file = assert(io.open(self.params.TrainingData, "r"), 'cannot open TrainingData')

    logger.info('Writing filtered data to %s', self.params.OutputFile)
    local output = assert(io.open(self.params.OutputFile, "w"), 'cannot open OutputFile')

    local score_file
    if self.params.save_score then
        logger.info('Writing score to %s', self.params.save_score_file)
        score_file = assert(io.open(self.params.save_score_file, 'w'), 'cannot open save_score_file')
    end

    local removed_file
    if self.params.save_removed then
        logger.info('Writing removed examples to %s', self.params.save_removed_file)
        removed_file = assert(io.open(self.params.save_removed_file, 'w'), 'cannot open save_removed_file')
    end

    local num = 0
    while true do
        local line = train_file:read("*line")
        if line == nil then
            break
        end
        num = num + 1
        if score_file then
            score_file:write(self.score[num] .. '\n')
        end
        if self.remove_indexes[num] == nil then
            output:write(line .. "\n")
        elseif removed_file then
            removed_file:write(line .. "\n")
        end
    end

    output:close()
    if self.params.save_score then
        score_file:close()
    end
    if self.params.save_removed then
        removed_file:close()
    end
end

-- Compute the vector representation of the top responses. (TopResponseEmbedding)
function EncoderDistiller:ComputeTopResponse()
    logger.info('Computing embeddings for top responses...')

    local open_top_response_file = assert(io.open(self.params.TopResponseFile, "r"),
        'cannot open TopResponseFile')

    local End = 0
    -- A list to hold all top response embeddings.
    self.TopResponseEmbedding = torch.Tensor():cuda()

    while End == 0 do
        End, self.Word_s, self.Word_t,
        self.Mask_s, self.Mask_t,
        self.Left_s, self.Left_t,
        self.Padding_s, self.Padding_t = self.dataset:read_train(open_top_response_file)

        if End == 1 then
            break
        end

        -- in decoding mode, only source is considered. target is not important.
        self.mode = "decoding"
        self.Word_s = self.Word_s:cuda()
        self.Padding_s = self.Padding_s:cuda()
        -- Use the encoder output as embedding.
        self:model_forward()

        local embed = torch.Tensor(self.last[2 * self.params.layers - 1]:size()):cuda():copy(self.last[2 * self.params.layers - 1])

        -- collect in a list.
        if self.TopResponseEmbedding:nDimension() == 0 then
            -- initial case: empty list.
            self.TopResponseEmbedding = embed
        else
            -- append to the list.
            self.TopResponseEmbedding = torch.cat(self.TopResponseEmbedding, embed, 1)
        end
    end
    -- Normalize each vector, which lives in Dim-2.
    self.TopResponseEmbedding = nn.Normalize(2):cuda():forward(self.TopResponseEmbedding)
end

return EncoderDistiller
