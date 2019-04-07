local FREQ_THRESHOLD = 100

local function load_dictionary(path)
    local file = assert(io.open(path), 'cannot open dictionary file')
    local dict = {}
    local count = 1
    while true do
        local line = file:read('*line')
        if not line then
            break
        end
        line = stringx.strip(line)
        dict[line] = count
        count = count + 1
    end
    file:close()
    return dict
end

local function load_response_file(path)
    local file = assert(io.open(path), 'cannot open response file')
    local responses = {}
    while true do
        local line = file:read('*line')
        if not line then
            break
        end
        table.insert(responses, stringx.split(line, '|')[2])
    end
    return responses
end

local function count_occurrence(lines)
    local counter = {}
    for _, line in ipairs(lines) do
        local value = (counter[line] == nil) and 0 or counter[line]
        counter[line] = value + 1
    end
    return counter
end

local function find_generic(stat, dict)
    local function to_numbers(line)
        local ids = {}
        for _, str in ipairs(stringx.split(line, ' ')) do
            table.insert(ids, dict[str])
        end
        return stringx.join(' ', ids)
    end

    local output = {}
    for line, freq in pairs(stat) do
        if freq >= FREQ_THRESHOLD then
            local ids = to_numbers(line)
            table.insert(output, { ids, freq })
        end
    end
    return output
end

local function extract_top(input, output, dict)
    local dict = load_dictionary(dict)
    local lines = load_response_file(input)
    local stat = count_occurrence(lines)
    local generic = find_generic(stat, dict)

    output = assert(io.open(output, 'w'), 'cannot open output')
    for _, data in ipairs(generic) do
        output:write(string.format('%s|%s\n', data[1], data[2]))
    end
    output:close()
end

return extract_top
