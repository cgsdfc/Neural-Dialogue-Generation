local function shuffle(a)
    local n = #a
    local t
    local k
    while (n > 0) do
        t = a[n]
        k = math.random(n)
        a[n] = a[k]
        a[k] = t
        n = n - 1
    end
    return a
end

local function shuffle_dataset(filename)
    local data = {}
    local file = io.open(filename)
    while true do
        local line = file:read('*line')
        if not line then
            break
        end
        table.insert(data, line)
    end
    file:close()
    shuffle(data)
    file = io.open(filename, 'w')
    for _, line in ipairs(data) do
        file:write(line .. '\n')
    end
    file:close()
end

return shuffle_dataset
