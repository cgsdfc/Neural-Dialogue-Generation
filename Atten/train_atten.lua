require "fbtorch"
require "cunn"
require "cutorch"
require "nngraph"

-- Parse command line.
local params = torch.reload("./parse")
-- Create a model.
local model = torch.reload("./atten");

cutorch.manualSeed(123)
cutorch.setDevice(params.gpu_index)
model:Initial(params)
model:train()
