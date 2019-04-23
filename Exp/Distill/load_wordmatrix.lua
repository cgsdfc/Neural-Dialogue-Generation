local function load_wordmatrix()
    local filename = 'data/wordmatrix_movie_25000'
    local file = torch.DiskFile(filename):binary()
    local obj = file:readObject()
    file:close()
    return obj
end

local wordmatrix = load_wordmatrix()
print('size():', wordmatrix:size())
print('[1]:', wordmatrix[1])

return wordmatrix
