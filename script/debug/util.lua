local util = {}

-- reload: unload a module and re-require it
local function reload(mod, ...)
    package.loaded[mod] = nil
    return require(mod, ...)
end

util.reload = reload
return util