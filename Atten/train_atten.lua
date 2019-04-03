require "torch"
require "cunn"
require "cutorch"
require "nngraph"

-- reload: unload a module and re-require it
local function reload(mod, ...)
    package.loaded[mod] = nil
    return require(mod, ...)
end

-- Parse command line.
local params = reload("./parse")
-- Create a model.
local model = reload("./atten");

cutorch.manualSeed(123)
cutorch.setDevice(params.gpu_index)
model:Initial(params)
model:train()
