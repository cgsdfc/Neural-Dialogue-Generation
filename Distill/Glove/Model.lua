require "cutorch"
require "nn"
require 'cunn'
require "nngraph"
require 'logroll'

local GloveDistiller = torch.class('GloveDistiller')
local logger = logroll.print_logger()

function GloveDistiller:MakeFilenames()
    assert(path.isdir(self.params.saveFolder), 'saveFolder does not exist!')
    local basename = path.basename(self.params.TrainingData)
    self.output_file = path.join(self.params.saveFolder, basename)
    if self.params.save_summary then
        self.summary_file = path.join(self.params.saveFolder, basename .. '_summary.csv')
    end
end

function GloveDistiller:BuildEmbeddingLayer()
    logger.info('loading Glove embeddings from %s', self.params.WordMatrix)
    local file = torch.DiskFile(self.params.WordMatrix, "r"):binary()
    local embedding = file:readObject()
    file:close()

    logger.info('vocab_size: %d', embedding:size(1))
    logger.info('vector_dimension: %d', (embedding:size(2)))

    logger.info('creating LookupTable layer from it')
    self.LookUpTable = nn.LookupTable(embedding:size()):cuda()
    local parameter = self.LookUpTable:parameters()
    parameter[1]:copy(embedding:cuda())

    logger.info('Creating feed-forward network getMatrix')
    self.getMatrix = nn.Sequential()
    self.getMatrix:add(self.LookUpTable)
    logger.info('Using the sum of word embeddings as utterance embedding')
    self.getMatrix:add(nn.Sum(2))
    self.getMatrix:add(nn.Normalize(2))
    self.getMatrix = self.getMatrix:cuda()
end

function GloveDistiller:__init(params)
    self.params = params
    self:MakeFilenames()

    logger.info('Building Embedding Layer')
    self:BuildEmbeddingLayer()

    logger.info('Loading four-gram')
    self:LoadFourGram()

    logger.info('Loading lines of top response from %s', self.params.TopResponseFile)
    self.top_response_lines = self:ReadFile(self.params.TopResponseFile)
    if #self.top_response_lines == 0 then
        logger.error('top_response_file is empty: %s', self.params.TopResponseFile)
        error('Distillation ends')
    end

    logger.info('Computing top response embeddings')
    self.top_response_embedding = self:lines2Embedding(self.top_response_lines)
end

function GloveDistiller:lines2Embedding(lines)
    local max_length = -100 -- max length of lines
    local All_tensors = {}

    for i, str in pairs(lines) do
        local split = stringx.split(str, " ")
        if #split > max_length then
            max_length = #split
        end
        -- Convert a list of number strings to a tensor:
        -- ['1','2'] => Tensor({1,2})
        local tensor = torch.Tensor(1, #split):zero()
        for j = 1, #split do
            tensor[1][j] = tonumber(split[j])
        end
        All_tensors[#All_tensors + 1] = tensor
    end

    local matrix = torch.Tensor(#lines, max_length):fill(1)
    for i, tensor in pairs(All_tensors) do
        matrix:sub(i, i, 1, tensor:size(2)):copy(tensor)
    end

    local vector = self.getMatrix:forward(matrix)
    return torch.Tensor(vector:size()):copy(vector):cuda()
end

-- Read the Top Response File. Return a table of 4-grams.
function GloveDistiller:LoadFourGram()
    local top_response_file = assert(io.open(self.params.TopResponseFile, "r"),
        'cannot open TopResponseFile')
    self.FourGram = {}

    while true do
        local line = top_response_file:read("*line")
        if line == nil then
            break
        end
        local t = line:find("|")
        -- Take the thing *after the bar* as top response.
        line = line:sub(t + 1, -1)
        local G = stringx.split(line, " ")
        if #G >= 4 then
            for i = 1, #G - 3 do
                local gram = G[i] .. " " .. G[i + 1] .. " " .. G[i + 2] .. " " .. G[i + 3]
                if self.FourGram[gram] == nil then
                    self.FourGram[gram] = 1
                end
            end
        end
    end
end

-- Read the TopResponseFile. Return a table of lines.
function GloveDistiller:ReadFile(filename)
    local file = assert(io.open(filename, "r"), 'cannot open file ' .. filename)
    local lines = {}
    while true do
        local line = file:read("*line")
        if line == nil then
            break
        end
        local t = line:find("|")
        -- The TOPRES is really after a bar.
        lines[#lines + 1] = line:sub(t + 1, -1)
    end
    file:close()
    return lines
end

-- Compute similarity score.
function GloveDistiller:GetScore()
    logger.info('Computing cosine similarity scores...')
    local open_train = assert(io.open(self.params.TrainingData), 'cannot open TrainingData')
    local current_lines = {}
    self.all_scores = torch.Tensor():cuda()
    local num_lines = 0

    -- Instead of using a Dataset to get batches, here we use a buffer -- current_lines.
    while true do
        local line = open_train:read("*line")
        if line == nil then
            break
        end

        local splits = stringx.split(line, "|")
        local str = stringx.strip(splits[2])

        current_lines[#current_lines + 1] = str
        num_lines = num_lines + 1

        if #current_lines % self.params.batch_size == 0 then
            local current_matrix = self:lines2Embedding(current_lines)
            local score = nn.MM(false, true):cuda():forward({ current_matrix, self.top_response_embedding })
            score = torch.max(score, 2)

            if self.all_scores:nDimension() == 0 then
                self.all_scores = score
            else
                self.all_scores = torch.cat(self.all_scores, score, 1)
            end

            current_lines = {}
        end
    end

    assert(num_lines >= self.params.batch_size, '#Examples is smaller than batch_size, no score is computed!')
    self.all_scores = torch.reshape(self.all_scores, self.all_scores:size(1))
    self.total_lines = num_lines
end

function GloveDistiller:RemoveFiles()
    logger.info('Removing examples...')
    local output = assert(io.open(self.output_file, "w"), 'cannot open output_file')
    local summary
    if self.summary_file then
        summary = torch.DiskFile(self.summary_file, 'w')
        summary:writeString('Score,Distilled,Example\n')
    end

    logger.info('Ranking training examples on their scores')
    local k = torch.floor(0.3 * self.all_scores:size(1))
    logger.info('k for topk: %d', k)

    local _, index = torch.topk(self.all_scores, k, true)
    local remove_indexes = {}

    local num_to_remove = torch.floor(self.total_lines * self.params.distill_rate)
    logger.info('number of examples to remove: %d', num_to_remove)

    for i = 1, num_to_remove do
        remove_indexes[index[i]] = 1
    end

    local num = 0
    local open_train = assert(io.open(self.params.TrainingData), 'cannot open TrainingData')
    local four_gram_distill_num = 0
    local cosine_distill_num = 0

    while true do
        local line = open_train:read("*line")
        if line == nil then
            break
        end

        num = num + 1
        if num > self.all_scores:size(1) then
            logger.error('number of examples exceeds that of all_scores: %d', num)
            break
        end

        local distill = false
        if remove_indexes[num] ~= nil then
            cosine_distill_num = cosine_distill_num + 1
            distill = true
        end

        local t = line:find("|")
        local target = line:sub(t + 1, -1)

        if self.params.distill_four_gram and not distill then
            local G = stringx.split(target, " ")
            for i = 1, #G - 3 do
                local gram = G[i] .. " " .. G[i + 1] .. " " .. G[i + 2] .. " " .. G[i + 3]
                if self.FourGram[gram] ~= nil then
                    distill = true
                    four_gram_distill_num = four_gram_distill_num + 1
                    break
                end
            end
        end

        local score = self.all_scores[num]
        if summary then
            summary:writeString(string.format('%f,%s,%s\n', score, distill, line))
        end

        if not distill then
            output:write(line .. "\n")
        end
    end
    logger.info('Distilled due to cosine similarity: %d', cosine_distill_num)
    logger.info('Distilled due to 4-grams: %d', four_gram_distill_num)

    output:close()
    if summary then
        summary:close()
    end
end

function GloveDistiller:Distill()
    self:GetScore()
    self:RemoveFiles()
end

return GloveDistiller
